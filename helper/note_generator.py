import itertools
import json
import os
import re
import datetime
from collections import Counter
from datetime import timedelta

from joblib import Parallel, delayed
from transformers import (
    AutoTokenizer, set_seed,
)

import numpy as np
import pandas as pd
import pickle

from helper.note_template import NoteTemplate

gender_categories = ['female', 'male']
race_categories = ['race not recorded', 'hispanic or latino', 'asian', 'black or african american',
                   'american indian or alaska native', 'white', 'native hawaiian or other pacific islander']
female_name = 'Mary Smith'
male_name = 'James Smith'
female_pronoun = 'she'
male_pronoun = 'he'

# Define overall template structure
template_note_structure = \
    """{summary}

    {visits}"""

summary_template = 'Summary: ${name} is a ${age} year old ${race_text} ${gender}.'
summary_template_config = {
    'defaults': {'name': 'The patient'},
    'fns': [lambda n: n.replace(' male', ' man').replace(' female', ' women')]
}

visit_template = "On ${start} ${name} ${type} ${specialty} ${duration} ${condition1}.${gender} " + \
                 " ".join(['${condition' + str(i) + '}' for i in range(2, 100)]) + "." + \
                 " A " + " ".join(['${procedure' + str(i) + '}' for i in range(1, 100)]) + " was performed."
visit_template_config = {
    'prefixes':
        dict({'duration': 'for ',
              'specialty': 'at ',
              'condition1': 'with a primary complaint of ',
              'condition2': 'The patient was also treated for '},
             **({'condition' + str(i): ' and ' for i in range(3, 100)}),
             **({'procedure' + str(i): ' and ' for i in range(2, 100)})),
    'suffixes': {'duration': ' days',
                 **({'procedure' + str(i): '#p#' for i in range(2, 100)})},
    'defaults': {'type': 'visit', 'name': 'the patient', 'pronoun': 'The patient', 'gender': ''},
    'pre': {
        'duration': lambda td: None if (td is None or str(td) == '' or td.days == 0) else td,
        # Categories: Pharmacy visit, Office Visit, Hospice, Inpatient Visit, Outpatient Visit
        'type': lambda t: 'stayed in the hospital' if t.lower() == 'inpatient visit' else
        ('visited the hospital' if t.lower() == 'outpatient visit' else
         ('stayed in the hospice' if t.lower() == 'hospice' else
          ('saw a doctor' if t.lower() == 'office visit' else
           'visited a pharmacy' if t.lower() == 'pharmacy visit' else t))),
        'specialty': lambda s: None if not pd.isna(s) and s.lower() == 'no matching concept' else s},
    'fns': [lambda n: n.replace('.male The patient', '. He').replace('.female The patient', '. She'),
            lambda n: n.replace('.male ', '. ').replace('.female ', '. '),
            lambda n: n.replace('saw a doctor at', 'saw a doctor for').replace('visited a pharmacy at',
                                                                               'visited a pharmacy for'),
            lambda n: n.replace('A' + (' ' * 100) + 'was performed.', ''),
            lambda n: re.sub(r"\#p\#\s+was performed.", ' were performed.', n),
            lambda n: n.replace('#p#', '')]
}

visit_list_template = "${start}: ${type} ${specialty} ${duration}" + \
                      " ".join(['${condition' + str(i) + '}' for i in range(1, 100)]) + "" + \
                      " ".join(['${procedure' + str(i) + '}' for i in range(1, 100)]) + ""
visit_list_template_config = {
    'prefixes':
        dict({'duration': 'for ',
              'specialty': 'at ',
              'condition1': '\nConditions:\n- ',
              'procedure1': '\nProcedures:\n- '},
             **({'condition' + str(i): '\n- ' for i in range(2, 100)}),
             **({'procedure' + str(i): '\n- ' for i in range(2, 100)})),
    'suffixes': {'duration': ' days'},
    'defaults': {'type': 'visit', 'name': 'the patient', 'pronoun': 'The patient', 'gender': ''},
    'pre': {
        'duration': lambda td: None if (td is None or str(td) == '' or td.days == 0) else td,
        # Categories: Pharmacy visit, Office Visit, Hospice, Inpatient Visit, Outpatient Visit
        'type': lambda t: 'stayed in the hospital' if t.lower() == 'inpatient visit' else
        ('visited the hospital' if t.lower() == 'outpatient visit' else
         ('stayed in the hospice' if t.lower() == 'hospice' else
          ('saw a doctor' if t.lower() == 'office visit' else
           'visited a pharmacy' if t.lower() == 'pharmacy visit' else t))),
        'specialty': lambda s: None if not pd.isna(s) and s.lower() == 'no matching concept' else s},
    'fns': [lambda n: n.replace('.male ', '. ').replace('.female ', '. '),
            lambda n: n.replace('saw a doctor at', 'saw a doctor for').replace('visited a pharmacy at',
                                                                               'visited a pharmacy for')]
}


class NoteGenerator:
    def __init__(self, task, data, training_end_date="2020-10-01", feature_weights_list=None, template_suffix='',
                 zero_shot_weights=None, concept_names=(None, None)):
        """ Parse data of a OMOP cohort into class structures. """
        self.person_ids = [person['person_id'] for person in data]  # correct ordering of persons
        self.visits = None
        self.non_temporal = None
        self.temporal = None
        self.visits, self.non_temporal, self.temporal = self.parse_data(data)
        assert set(self.person_ids) == set(self.visits['person_id'].tolist())

        # Load id - value mapping
        with open('/root/omop-pkg/misc/permute_concepts/id_value_map_eol_loh_surgery.txt', 'r') as file:
            self.values = {int(k): v for k, v in (json.load(file)).items()}
        if feature_weights_list is None or zero_shot_weights is not None:
            feature_weights_list = pd.DataFrame([], columns=['concept_id', 'interval', 'weight'])
        # Necessary to assign feature weights with intervals
        self.training_end_date = datetime.datetime.strptime(training_end_date, '%Y-%m-%d')
        self.task = task
        self.template_suffix = template_suffix

        self.shortened_conditions = read_shortened_concepts(concept_names[0])
        self.shortened_procedures = read_shortened_concepts(concept_names[1])

        # Read dictionaries for permuting concepts
        def read_permuted_concepts(file_name):
            with open('/root/omop-pkg/misc/permute_concepts/' + file_name) as f:
                return eval(f.readline())
        self.permuted_conditions = read_permuted_concepts('condition_ids_permuted.txt')
        self.permuted_procedures = read_permuted_concepts('procedure_ids_permuted.txt')

        # Pandas debug settings
        # pd.set_option('display.max_rows', 500)
        # pd.set_option('display.max_columns', 500)
        # pd.set_option('display.width', 500)

        # Create unique temporal concepts for each visit
        self.temporal = self.temporal.drop_duplicates(ignore_index=True)
        # If numbered conditions exist, prefer them over non-numbered
        self.temporal.loc[self.temporal['concept'] == 'condition', 'concept'] = 'condition none'
        self.temporal = pd.concat([self.temporal.loc[~(self.temporal['concept'].str.startswith('condition'))],
                                   self.temporal.loc[self.temporal['concept'].str.startswith('condition')].
                                  sort_values('concept').drop_duplicates(
                                       [c for c in self.temporal.columns if c != 'concept'], keep='first')])
        self.temporal.sort_values(['person_id', 'visit_id', 'concept'], inplace=True)

        # Read feature weights and assign them to concepts
        # feat_name_regex = re.compile(r"(\d+) - (condition|procedure|drug) - (.+) - (\d+) days")
        self.feature_weights = feature_weights_list
        # Distinguish feature weight file with person specific weights
        if 'person_id' in feature_weights_list.columns:
            self.feature_weights['person_id'] = self.feature_weights['person_id'].astype(int)
            assert set(self.person_ids) == set(self.feature_weights['person_id'].tolist())
        self.feature_weights['concept_id'] = self.feature_weights['concept_id'].astype(int)
        self.feature_weights['interval'] = pd.to_timedelta(pd.to_numeric(self.feature_weights['interval']), unit='days')
        self.feature_weights['interval_start'] = self.training_end_date - self.feature_weights['interval']
        self.feature_weights['abs_weight'] = abs(self.feature_weights['weight'])

        if not self.feature_weights.empty:
            # Assign feature weights to visits when they occur in given interval
            original_columns = self.visits.columns.tolist()
            original_size = self.visits.shape[0]
            # Assign values to concept_ids temporarily bc visit entries have no concept_id
            self.feature_weights.loc[self.feature_weights['concept_id'] == 581458, 'value'] = 'Pharmacy visit'
            self.feature_weights.loc[self.feature_weights['concept_id'] == 581477, 'value'] = 'Office Visit'
            self.feature_weights.loc[self.feature_weights['concept_id'] == 8546, 'value'] = 'Hospice'
            self.feature_weights.loc[self.feature_weights['concept_id'] == 9201, 'value'] = 'Inpatient Visit'
            self.feature_weights.loc[self.feature_weights['concept_id'] == 9202, 'value'] = 'Outpatient Visit'
            # Distinguish feature weight file with person specific weights
            if 'person_id' not in feature_weights_list.columns:
                self.visits = self.visits.merge(self.feature_weights.loc[self.feature_weights['concept_id'].isin([581458, 581477, 8546, 9201, 9202])],
                                                how='left', left_on='type', right_on='value')
            else:
                self.visits = self.visits.merge(self.feature_weights.loc[self.feature_weights['concept_id'].isin([581458, 581477, 8546, 9201, 9202])],
                                                how='left', left_on=['type', 'person_id'], right_on=['value', 'person_id'])
                assert set(self.visits['person_id'].tolist()) == set(self.feature_weights['person_id'].tolist())
            self.feature_weights = self.feature_weights.drop(columns=['value'])
            self.visits[['weight', 'abs_weight']] = self.visits[['weight', 'abs_weight']].fillna(0.)
            # Set all weights for visits before interval to zero
            self.visits.loc[self.visits['start'] < self.visits['interval_start'], ['weight', 'abs_weight']] = 0
            # Only keep weight of earliest visit (also ignore interval now)
            self.visits.sort_values('start', ascending=False, inplace=True)
            self.visits['num_visits_per_category'] = self.visits.groupby(['person_id', 'value']).cumcount()
            self.visits.loc[self.visits['num_visits_per_category'] != 0, ['weight', 'abs_weight']] = 0
            self.visits = self.visits.drop(columns=['num_visits_per_category'])

            # Sum all interval weights together.
            self.visits = self.visits[original_columns + ['weight', 'abs_weight']] \
                .groupby(original_columns, dropna=False)[['weight', 'abs_weight']].sum().reset_index()
            # Lowercase specialties
            self.visits['specialty'] = self.visits['specialty'].str.lower()
            assert original_size == self.visits.shape[0]
        else:
            self.visits[['weight', 'abs_weight']] = 0

        original_columns = self.temporal.columns.tolist()
        original_size = self.temporal.shape[0]
        if not self.feature_weights.empty:
            # Assign feature weights to concepts when they occur in the given interval
            self.temporal = self.temporal.merge(self.visits[['id', 'start']], how='left', left_on='visit_id', right_on='id')
            # Distinguish feature weight file with person specific weights
            if 'person_id' not in feature_weights_list.columns:
                self.temporal = self.temporal.merge(
                    self.feature_weights.loc[:, [c for c in self.feature_weights.columns if c not in ['concept', 'value']]],
                    how='left', on='concept_id')
            else:
                self.temporal = self.temporal.merge(
                    self.feature_weights.loc[:, [c for c in self.feature_weights.columns if c not in ['concept', 'value']]],
                    how='left', on=['concept_id', 'person_id'])
            self.temporal[['weight', 'abs_weight']] = self.temporal[['weight', 'abs_weight']].fillna(0.)
            # Set all weights for visits before interval to zero and sum all interval weights together.
            self.temporal.loc[self.temporal['start'] < self.temporal['interval_start'], ['weight', 'abs_weight']] = 0
            self.temporal = self.temporal[original_columns + ['weight', 'abs_weight']] \
                .groupby(original_columns)[['weight', 'abs_weight']].sum().reset_index()
            assert original_size == self.temporal.shape[0]
        else:
            self.temporal[['weight', 'abs_weight']] = 0
            self.temporal = self.temporal.merge(self.visits[['id', 'start']], how='left', left_on='visit_id', right_on='id')

            # Optionally if specific zero shot weighting given use this schema
            if zero_shot_weights.startswith('most_frequent'):
                self.temporal['weight'] = self.temporal.groupby(['person_id', 'concept_id'])['concept_id'].transform('count')
            elif zero_shot_weights.startswith('least_frequent'):
                self.temporal['weight'] = 10000 - self.temporal.groupby(['person_id', 'concept_id'])['concept_id'].transform('count')
            elif zero_shot_weights.startswith('oldest'):
                self.temporal['weight'] = (self.training_end_date - self.temporal['start']).dt.days
            elif zero_shot_weights.startswith('recent'):
                self.temporal['weight'] = 10000 - (self.training_end_date - self.temporal['start']).dt.days
            # Only keep selected concepts
            if zero_shot_weights.endswith('_conditions'):
                self.temporal.loc[self.temporal['concept'].str.startswith('procedure'), 'weight'] = 0.
            if zero_shot_weights.endswith('_procedures'):
                self.temporal.loc[self.temporal['concept'].str.startswith('condition'), 'weight'] = 0.

            self.temporal['abs_weight'] = abs(self.temporal['weight'])
            self.temporal = self.temporal[original_columns + ['weight', 'abs_weight']]

        # Create unique concept numbering. First criterion existing numbering, second weight, third concept_id.
        self.temporal = self.temporal.sort_values(by=['person_id', 'visit_id', 'concept', 'weight', 'concept_id'],
                                                  ascending=[True, True, True, False, True])
        # Columns concept_id and abs_weight not necessary here
        self.temporal.loc[
            self.temporal['concept'].str.startswith('condition'),
            'concept'] = 'condition' + (self.temporal.loc[self.temporal['concept'].str.startswith('condition')]
                                        .groupby(by=['person_id', 'visit_id'], sort=False).cumcount() + 1).apply(str)
        self.temporal.sort_values(['person_id', 'visit_id', 'concept'], inplace=True)
        assert original_size == self.temporal.shape[0]

        # Lowercase first letter of concept values
        self.values = {k: lower_first_char_if_second_low(v) for k, v in self.values.items()}

        # Add additional non temporal variables
        self.non_temporal.rename({'Age at end_date': 'age'}, axis=1, inplace=True)
        self.non_temporal['gender'] = [gender_categories[i] for i in self.non_temporal['Gender M(1)/F(0)'].to_list()]
        self.non_temporal['race_text'] = [race_categories[i] if race_categories[i] != 'race not recorded' else None
                                          for i in self.non_temporal['Race'].to_list()]

        # Debug: Output concept renaming
        # def output_concept_name_statistics(x):
        #     collect = []
        #     for concept_id in self.temporal['concept_id'].tolist():
        #         if concept_id in self.values.keys():
        #             if self.values[concept_id] in x.keys():
        #                 collect.append((self.values[concept_id], x[self.values[concept_id]]))
        #     c = list(Counter(collect).items())
        #     c.sort(key=lambda x: x[1], reverse=True)
        #     for i in range(0, min(100, len(c))):
        #         print(c[i][1], c[i][0][0], '--->', c[i][0][1])

        # output_concept_name_statistics(read_shortened_concepts('conditions_eol_loh_surgery_short.txt'))
        # output_concept_name_statistics(read_shortened_concepts('procedures_eol_loh_surgery_short.txt'))

    def permute_concept(self, mapping, concept_id):
        if pd.isna(concept_id):
            return concept_id
        concept_id = int(concept_id)
        if mapping is not None:
            if concept_id in mapping.keys():
                return mapping[concept_id]
            else:
                # fallback return same concept, but should usually never happen.
                print("Concept not found in permutation mapping.")
                return concept_id
        else:
            return concept_id

    def generate_notes(self, method, **kwargs):
        # Call method generate_notes + method
        func = getattr(NoteGenerator, 'generate_notes_' + method)
        return func(self, **kwargs)

    def generate_notes_n_important_visits(self, **kwargs):
        return self.generate_notes_n_important_concepts(incl_conditions=False, incl_procedures=False, **kwargs)

    def generate_notes_n_important_visits_and_conditions(self, **kwargs):
        return self.generate_notes_n_important_concepts(incl_conditions=True, incl_procedures=False, **kwargs)

    def generate_notes_n_important_visits_and_conditions_and_procedures(self, **kwargs):
        return self.generate_notes_n_important_concepts(incl_conditions=True, incl_procedures=True, **kwargs)

    def generate_notes_list_n_important_visits_and_conditions_and_procedures(self, **kwargs):
        return self.generate_notes_n_important_concepts(incl_conditions=True, incl_procedures=True, create_list=True,
                                                        **kwargs)

    def generate_notes_list_permuted_n_important_visits_and_conditions_and_procedures(self, **kwargs):
        return self.generate_notes_n_important_concepts(incl_conditions=True, incl_procedures=True, create_list=True,
                                                        permuted=True, **kwargs)

    def generate_notes_permuted_n_important_visits_and_conditions_and_procedures(self, **kwargs):
        return self.generate_notes_n_important_concepts(incl_conditions=True, incl_procedures=True, permuted=True,
                                                        **kwargs)

    def generate_notes_n_important_concepts(self, **kwargs):
        max_token_length = kwargs.get('max_token_length', 1024)
        tokenizer_model = "bigscience/T0pp"
        max_token_length_safety_margin = 0.95
        num_concepts = kwargs.get('num_concepts', None)
        incl_zero_weight = kwargs.get('incl_zero_weight', False)
        abs_weight = kwargs.get('abs_weight', True)
        incl_conditions = kwargs.get('incl_conditions', False)
        incl_procedures = kwargs.get('incl_procedures', False)
        create_list = kwargs.get('create_list', False)
        permuted = kwargs.get('permuted', False)

        # For 10 concepts token limit probably not relevant.
        if num_concepts > 10:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        else:
            tokenizer = lambda x, **kw: {'input_ids': []}

        def token_length(text):
            tokenized = tokenizer(text, padding=False, truncation=True, add_special_tokens=True, max_length=1024)['input_ids']
            return len(tokenized)

        summary_note_temp = NoteTemplate(summary_template, **summary_template_config)
        if create_list:
            visit_note_temp = NoteTemplate(visit_list_template, **visit_list_template_config)
        else:
            visit_note_temp = NoteTemplate(visit_template, **visit_template_config)
        visits = self.visits.copy()
        conditions = self.temporal.loc[self.temporal['concept'].str.startswith('condition')].copy()
        self.set_absolute_weights(abs_weight, conditions, visits)
        if incl_procedures:
            procedures = self.temporal.loc[self.temporal['concept'].str.startswith('procedure')].copy()
            self.set_absolute_weights(abs_weight, procedures)
        else:
            procedures = pd.DataFrame([], columns=['person_id', 'id', 'type', 'start', 'duration', 'specialty',
                                                   'weight', 'abs_weight', 'visit_id'])

        if not incl_zero_weight:
            conditions = conditions.loc[conditions['weight'] != 0]
            procedures = procedures.loc[procedures['weight'] != 0]

        # Select first occurrence of each concept, since non-zero removed, the earliest non-zero concept remains.
        conditions = self.select_first_concept_occurrence(conditions, visits)
        if incl_procedures:
            procedures = self.select_first_concept_occurrence(procedures, visits)

        # Add weight of first condition to each visit because they always go together and remove condition1 entries
        original_columns = visits.columns
        visits = visits.merge(conditions.loc[conditions['concept'] == 'condition1', ['visit_id', 'weight']],
                              how='left', left_on='id', right_on='visit_id')
        visits['weight'] = visits['weight_x'] + (visits['weight_y'].fillna(0.))
        visits = visits[original_columns]
        visits = self.add_temporal_value_to_visits(visits, self.temporal, 'condition1')
        conditions = conditions.loc[conditions['concept'] != 'condition1']

        if not incl_conditions:
            conditions = pd.DataFrame([], columns=['person_id', 'id', 'type', 'start', 'duration', 'specialty',
                                                   'weight', 'abs_weight'])

        # Optionally shuffle concepts and shorten concepts
        if permuted:
            visits['condition1'] = visits['condition1'].apply(lambda x: self.permute_concept(self.permuted_conditions, x))
            conditions['concept_id'] = conditions['concept_id'].apply(lambda x: self.permute_concept(self.permuted_conditions, x))
            procedures['concept_id'] = procedures['concept_id'].apply(lambda x: self.permute_concept(self.permuted_procedures, x))
        # Shorten concepts
        for concept_id, concept_value in self.values.items():
            if concept_value.lower() in self.shortened_conditions.keys():
                self.values[concept_id] = self.shortened_conditions[concept_value.lower()]
            elif concept_value.lower() in self.shortened_procedures.keys():
                self.values[concept_id] = self.shortened_procedures[concept_value.lower()]

        # Prefer more recent concepts
        visits.sort_values('start', ascending=False, inplace=True, ignore_index=True)
        conditions.sort_values('start', ascending=False, inplace=True, ignore_index=True)
        procedures.sort_values('start', ascending=False, inplace=True, ignore_index=True)
        # Split up visit, conditions, and procedures by patient to parallelize
        person_visits, person_conditions, person_procedures, rel_entities = {}, {}, {}, {}
        for p in self.person_ids:
            person_visits[p] = (visits.loc[visits['person_id'] == p].copy()).reset_index(drop=True)
            visits.drop(visits[visits['person_id'] == p].index, inplace=True)
            person_conditions[p] = (conditions.loc[conditions['person_id'] == p].copy()).reset_index(drop=True)
            conditions.drop(conditions[conditions['person_id'] == p].index, inplace=True)
            person_procedures[p] = (procedures.loc[procedures['person_id'] == p].copy()).reset_index(drop=True)
            procedures.drop(procedures[procedures['person_id'] == p].index, inplace=True)

            # Combine relevant concepts for each patient
            person_visits[p]['idx_col'] = person_visits[p].index
            person_conditions[p]['idx_col'] = person_conditions[p].index
            person_procedures[p]['idx_col'] = person_procedures[p].index

            def set_types(l, c):
                return list(map(lambda x: [c, int(x[1]), x[2]], l))

            rel_entities[p] = set_types((person_visits[p][['person_id', 'idx_col', 'weight']]).values.tolist(), 'v') +\
                set_types((person_conditions[p][['person_id', 'idx_col', 'weight']]).values.tolist(), 'c') +\
                set_types((person_procedures[p][['person_id', 'idx_col', 'weight']]).values.tolist(), 'p')
            # Have to do it again for visits here
            if not incl_zero_weight:
                rel_entities[p] = [x for x in rel_entities[p] if x[2] != 0]

        def create_note(p, p_relevant_entities, p_visits, p_conditions, p_procedures):
            # if n % 1000 == 0:
            #     print(f"Processed {n}/{len(self.person_ids)} persons.")
            note = template_note_structure
            non_temporary_info = self.non_temporal.loc[self.non_temporal['person_id'] == p].squeeze()
            name_pronoun_gender = self.derive_name_pronoun_series(non_temporary_info['gender'])
            s = summary_note_temp.substitute(non_temporary_info.combine_first(name_pronoun_gender))
            note = note.replace('{summary}', s)

            # Collect relevant concepts that end up in note
            selected_concepts = {}
            selected_concepts['age'] = int(non_temporary_info['age'])
            selected_concepts['sex_male'] = int(non_temporary_info['gender'] == 'male')
            selected_concepts['sex_female'] = int(non_temporary_info['gender'] != 'male')
            selected_concepts['race_' + (race_categories[int(non_temporary_info['Race'])]).replace(' ', '_')] = 1

            # Sort by start to prefer later occurrences
            p_relevant_entities.sort(reverse=True, key=lambda x: x[2])
            # Use these sorted visit ids to create a note with entries sorted by occurrence
            visits_idx_sorted_by_earliest_start = (p_visits.loc[p_visits['person_id'] == p].index.tolist())[::-1]
            included_visit_idx = set()
            conditions_per_visit_idx = {}
            procedures_per_visit_idx = {}
            sentence_per_visit_idx = {}
            sorted_visit_strings = []
            curr_note = self.clean_note(note.replace('{visits}', ''))
            for i in range(0, min(len(p_relevant_entities), num_concepts)):
                idx = p_relevant_entities[i][1]
                if p_relevant_entities[i][0] == 'v':
                    # Next most important entity is a visit
                    visit_idx = p_relevant_entities[i][1]
                    included_visit_idx.add(p_relevant_entities[i][1])
                elif p_relevant_entities[i][0] == 'c':
                    # Next most important entity is a concept
                    max_c_visit_idx = p_visits.index[p_visits['id'] == p_conditions.loc[idx, 'visit_id']][0]
                    visit_idx = max_c_visit_idx
                    included_visit_idx.add(max_c_visit_idx)
                    curr = conditions_per_visit_idx.get(max_c_visit_idx, pd.Series([], dtype=str))
                    conditions_per_visit_idx[max_c_visit_idx] = \
                        curr.append(pd.Series(self.values[p_conditions.at[idx, 'concept_id']],
                                              index=['condition' + str(curr.size + 2)]))
                    selected_concepts[str(self.values[p_conditions.at[idx, 'concept_id']])] = 1
                elif p_relevant_entities[i][0] == 'p':
                    # Next most important entity is a procedure
                    max_c_visit_idx = p_visits.index[p_visits['id'] == p_procedures.loc[idx, 'visit_id']][0]
                    visit_idx = max_c_visit_idx
                    included_visit_idx.add(max_c_visit_idx)
                    curr = procedures_per_visit_idx.get(max_c_visit_idx, pd.Series([], dtype=str))
                    procedures_per_visit_idx[max_c_visit_idx] = \
                        curr.append(pd.Series(self.values[p_procedures.at[idx, 'concept_id']],
                                              index=['procedure' + str(curr.size + 1)]))
                    selected_concepts[str(self.values[p_procedures.at[idx, 'concept_id']],)] = 1

                # Update sentence for visit
                visit_series = p_visits.iloc[visit_idx].combine_first(name_pronoun_gender)
                visit_series['condition1'] = self.values[visit_series['condition1']] if not \
                    pd.isna(visit_series['condition1']) else visit_series['condition1']
                selected_concepts[str(visit_series['condition1'])] = 1
                if visit_idx in conditions_per_visit_idx.keys():
                    visit_series = visit_series.combine_first(conditions_per_visit_idx[visit_idx])
                if visit_idx in procedures_per_visit_idx.keys():
                    visit_series = visit_series.combine_first(procedures_per_visit_idx[visit_idx])
                # Generate sentence from visit series that contains all relevant concepts
                sentence_per_visit_idx[visit_idx] = visit_note_temp.substitute(visit_series)

                # Generate temporary note with for currently selected visits
                sorted_visit_strings = [sentence_per_visit_idx[k] for k in visits_idx_sorted_by_earliest_start
                                        if k in sentence_per_visit_idx.keys()]
                temp_note = self.clean_note(note.replace('{visits}', '\n\n'.join(sorted_visit_strings)))
                if token_length(temp_note + self.template_suffix) < max_token_length * max_token_length_safety_margin:
                    curr_note = temp_note
                else:
                    # print("Reached token limit!!!")
                    break

            # Debugging purpose
            # print(p, '\n', curr_note)
            # print(p, len(sorted_visit_strings))
            return (curr_note, selected_concepts)

        # notes = Parallel(n_jobs=30)(delayed(create_note)(p, rel_entities[p], person_visits[p], person_conditions[p],
        #                                                  person_procedures[p]) for p in self.person_ids)
        notes_and_onehot = [create_note(p, rel_entities[p], person_visits[p], person_conditions[p], person_procedures[p])
                            for p in self.person_ids]
        notes = [x[0] for x in notes_and_onehot]
        onehot = [x[1] for x in notes_and_onehot]
        # Output concepts that end up in note
        # with open('note_concepts_all_most_frequent_conditions-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.p', 'wb') as f:
        #     pickle.dump(onehot, f)
        return notes

    def generate_notes_label_leakage(self, **kwargs):
        labels = kwargs['labels']
        summary_note_temp = NoteTemplate(summary_template, **summary_template_config)
        notes = []

        for i, p in enumerate(self.person_ids):
            note = template_note_structure
            non_temporary_info = self.non_temporal.loc[self.non_temporal['person_id'] == p, ['age', 'gender']].squeeze()
            name_pronoun_gender = self.derive_name_pronoun_series(non_temporary_info['gender'])
            s = summary_note_temp.substitute(non_temporary_info.combine_first(name_pronoun_gender))
            leakage = 'has several severe illnesses and is in a very unhealthy condition' if labels[i] == 1 else \
                'has no complaints and is in a very healthy condition'
            s += ' ' + (name_pronoun_gender['pronoun'] if 'pronoun' in name_pronoun_gender.keys() else 'this patient') + \
                 ' ' + leakage + '.'
            note = note.replace('{summary}', s)
            note = note.replace('{visits}', '')
            notes.append(self.clean_note(note))

        return notes

    def generate_notes_all_non_zero_conditions(self, **kwargs):
        summary_note_temp = NoteTemplate(summary_template, **summary_template_config)
        notes = []

        concepts = self.temporal.loc[self.temporal['concept'].str.startswith('condition')].copy()
        self.set_absolute_weights(kwargs['abs_weight'], concepts)
        # Only keep non-zero concepts
        concepts = concepts.loc[concepts['weight'] > 0]
        # Select first occurrence of each concept, since non-zero removed, the earliest non-zero concept remains.
        concepts = self.select_first_concept_occurrence(concepts, self.visits)

        for p in self.person_ids:
            note = template_note_structure
            non_temporary_info = self.non_temporal.loc[self.non_temporal['person_id'] == p, ['age', 'gender']].squeeze()
            s = summary_note_temp.substitute(non_temporary_info)
            note = note.replace('{summary}', s)

            visit_strings = []
            for idx, c in concepts.loc[concepts['person_id'] == p].sort_values('start', ascending=True).iterrows():
                visit_strings.append('- ' + c['start'].strftime('%Y') + ': ' + c['value'])
            note = note.replace('{visits}', 'Conditions:\n' + '\n'.join(visit_strings))
            # Debugging purpose
            # print(p, '\n', self.clean_note(note))
            notes.append(self.clean_note(note))

        return notes

    def generate_notes_n_important_conditions(self, **kwargs):
        """ Select the num_concept most important concepts (conditions) and add their respective visit entries. """
        incl_zero_weight = False
        summary_note_temp = NoteTemplate(summary_template, **summary_template_config)
        visit_note_temp = NoteTemplate(visit_template, **visit_template_config)
        notes = []
        concepts = self.temporal.loc[self.temporal['concept'].str.startswith('condition')].copy()

        self.set_absolute_weights(kwargs['abs_weight'], concepts)
        if not incl_zero_weight:
            concepts = concepts.loc[concepts['weight'] != 0]

        # Select first occurrence of each concept, since non-zero removed, the earliest non-zero concept remains.
        concepts = self.select_first_concept_occurrence(concepts, self.visits)

        # Select the n most relevant concepts per person. If same weight (usually zero) use most recent.
        concepts = concepts.groupby('person_id').apply(
            lambda x: x.sort_values(['weight', 'start'], ascending=False).head(kwargs['num_concepts'])).reset_index(drop=True)
        # Save visit id and then delete all condition1 entries bc they are already in visit entry.
        visit_ids = concepts['visit_id'].unique().tolist()
        concepts = concepts.loc[concepts['concept'] != 'condition1']
        # Rename all remaining conditions again starting from 2
        concepts = concepts.sort_values(by=['person_id', 'visit_id', 'concept', 'weight', 'value'],
                                        ascending=[True, True, True, False, True])
        concepts.loc[concepts['concept'].str.startswith('condition'), 'concept'] = \
            'condition' + (concepts.loc[concepts['concept'].str.startswith('condition')]
                           .groupby(by=['person_id', 'visit_id'], sort=False).cumcount() + 2).apply(str)
        # Add all included visits and add primary condition
        included_visits = self.visits.loc[self.visits['id'].isin(visit_ids)].copy()
        included_visits = self.add_temporal_value_to_visits(included_visits, self.temporal, 'condition1')
        for p in self.person_ids:
            note = template_note_structure
            s = summary_note_temp.substitute(self.non_temporal.loc[self.non_temporal['person_id'] == p].squeeze())
            note = note.replace('{summary}', s)
            visit_strings = []
            # Add visits with a selected condition.
            for idx, v in included_visits.loc[included_visits['person_id'] == p].sort_values('start',
                                                                                             ascending=False).iterrows():
                # Add additional conditions to visit
                for jdx, c in concepts.loc[(concepts['visit_id'] == v['id']) &
                                           (concepts['concept'].str.startswith('condition'))].sort_values('concept').iterrows():
                    v[c['concept']] = c['value']
                visit_strings.append(visit_note_temp.substitute(v))
            visit_strings.reverse()
            note = note.replace('{visits}', '\n\n'.join(visit_strings))
            # Debugging purpose
            # note = str(p) + '\n' + note
            # print(self.clean_note(note))
            notes.append(self.clean_note(note))

        return notes

    def generate_notes_last_n_visits(self, **kwargs):
        """ Generate note for max_visits last visit entries. Add additional_complaints many complaints per visit. """
        additional_complaints = 0
        summary_note_temp = NoteTemplate(summary_template, **summary_template_config)
        visit_note_temp = NoteTemplate(visit_template, **visit_template_config)
        notes = []
        # Add unique conditions to each visit
        visits_complaints = self.visits.copy()
        # Add additional complaints
        for i in range(1, max(2, min(additional_complaints + 2, 4))):
            visits_complaints = self.add_temporal_value_to_visits(visits_complaints, self.temporal,
                                                                  'condition' + str(i))
        # Sort visits by start date
        visits_complaints.sort_values('start', ascending=False, inplace=True)
        for p in self.person_ids:
            note = template_note_structure
            s = summary_note_temp.substitute(self.non_temporal.loc[self.non_temporal['person_id'] == p].squeeze())
            note = note.replace('{summary}', s)
            visit_strings = []
            # Add visits with non-empty primary condition, and remove exact same visit entries
            for idx, v in visits_complaints.loc[(visits_complaints['person_id'] == p) &
                                                (~(visits_complaints['condition1'].isna()))].iterrows():
                if len(visit_strings) >= kwargs['num_concepts']:
                    break
                visit_note = visit_note_temp.substitute(v)
                if visit_note not in visit_strings:
                    visit_strings.append(visit_note)
            visit_strings.reverse()
            note = note.replace('{visits}', '\n\n'.join(visit_strings))
            # Debugging purpose
            # note = str(p) + '\n' + note
            # print(self.clean_note(note))
            notes.append(self.clean_note(note))
        return notes

    @staticmethod
    def derive_name_pronoun_series(gender):
        use_name_and_pronoun = False
        use_pronoun = True

        if use_name_and_pronoun:
            if gender == "female":
                return pd.Series({'name': female_name, 'pronoun': female_pronoun, 'gender': gender})
            else:
                return pd.Series({'name': male_name, 'pronoun': male_pronoun, 'gender': gender})
        elif use_pronoun:
            if gender == "female":
                return pd.Series({'pronoun': female_pronoun, 'gender': gender})
            else:
                return pd.Series({'pronoun': male_pronoun, 'gender': gender})
        else:
            return pd.Series({})

    @staticmethod
    def add_temporal_value_to_visits(visits, temporal, name):
        visits = visits.merge(temporal.loc[temporal['concept'] == name, ['visit_id', 'concept_id']],
                              how='left', left_on='id', right_on='visit_id')
        visits.drop('visit_id', axis=1, inplace=True)
        visits.rename(columns={'concept_id': name}, inplace=True)
        return visits

    @staticmethod
    def set_absolute_weights(abs_weight, concepts=None, visits=None):
        if abs_weight:
            if concepts is not None:
                concepts['weight'] = concepts['abs_weight']
            if visits is not None:
                visits['weight'] = visits['abs_weight']

        if concepts is not None:
            concepts.drop('abs_weight', axis=1, inplace=True)
        if visits is not None:
            visits.drop('abs_weight', axis=1, inplace=True)

    @staticmethod
    def select_first_concept_occurrence(concepts, visits):
        concepts = concepts.merge(visits[['id', 'start']], how='left', left_on='visit_id', right_on='id')
        concepts.drop('id', axis=1, inplace=True)
        concepts = concepts.sort_values(by='start', ascending=True) \
            .drop_duplicates(subset=['person_id', 'concept_id'], keep='first').reset_index(drop=True)
        return concepts

    @staticmethod
    def parse_data(dataset):
        """ Parse data into separate dataframes for visits, non-temporal and temporal dataframes. """
        # Parse non-temporal attributes
        non_temporal = (dataset.remove_columns(['visits', 'dates', 'tok_visits'])).to_pandas()
        # Columns to expect as attributes for each visit
        visit_data_columns = ['id', 'type', 'start', 'duration', 'specialty']
        temporal_data_columns = ['person_id', 'visit_id', 'concept_id', 'concept']

        # Define and compiles regexes
        visit_regex = re.compile(r"\d+ - visit (\w+) - (.+)")
        concept_regex = re.compile(r"(\d+) - (condition|condition \d+|procedure|drug) - (.+)")

        # Prepare dataset
        dataset = dataset.add_column('person_visits', [[]] * len(dataset))
        dataset = dataset.add_column('person_temporal', [[]] * len(dataset))

        def parse_visits_and_temporal(person):
            assert len(person['visits']) == len(person['dates'])
            # Go through all visit entries in reverse order because they can actually correspond to several visits.
            # Use the visit_id to merge same visits. Going through them in reverse order allows that the most
            # relevant visit (the first) can overwrite visit related data from later ones.
            person['dates'] = [pd.to_datetime(d) for d in person['dates']]
            person_visits = {}
            person_temporal = []
            for i, visit in reversed(list(enumerate(person['visits']))):
                visit_id = int((person['dates'][i] - person['dates'][i]
                                .replace(microsecond=0, second=0, minute=0, hour=0)) / timedelta(microseconds=1))
                if visit_id not in person_visits.keys():
                    # Add empty default entries in case they are not provided
                    person_visits[visit_id] = {k: None for k in visit_data_columns}
                visit_data = []
                for concept in visit:
                    if visit_regex.match(concept):
                        # Parse all visits to get visit data as tuples (attr, value)
                        visit_data.append(visit_regex.search(concept).groups())
                    elif concept_regex.match(concept):
                        # All other concepts
                        # Need to keep everything as string to make it hf dataset compatible
                        id_concept_value = list(concept_regex.search(concept).groups())
                        person_temporal.append([str(person['person_id']), str(visit_id)] + id_concept_value[0:2])
                    else:
                        raise ValueError(f"Unknown concept string: {concept}")

                # Earlier visits with the same id update this data dictionary of the visit
                person_visits[visit_id].update(dict(visit_data))
                person_visits[visit_id]["person_id"] = str(person['person_id'])

            # Parse visit data into list format with fixed ordering
            person['person_visits'] = [[visit_dict[k] for k in (['person_id'] + visit_data_columns)]
                                       for visit_dict in person_visits.values()]
            person['person_temporal'] = person_temporal
            return person

        # dataset = [parse_visits_and_temporal(ex) for ex in dataset]
        dataset = dataset.map(parse_visits_and_temporal)

        # Create dataframe from list of lists of visit data / temporal data for each person
        temporal = pd.DataFrame(itertools.chain.from_iterable(dataset['person_temporal']), columns=temporal_data_columns)
        temporal[['person_id', 'visit_id', 'concept_id']] = temporal[['person_id', 'visit_id', 'concept_id']].apply(pd.to_numeric)
        visits = pd.DataFrame(itertools.chain.from_iterable(dataset['person_visits']), columns=['person_id'] + visit_data_columns)
        dataset = dataset.remove_columns(['person_visits', 'person_temporal'])

        # There is an edge case where concepts before inclusion date but visit after, so visit id not included.
        old_num_visits = visits.shape[0]
        visits = visits[visits['id'].notna()]
        if old_num_visits - visits.shape[0] > 0:
            print(f"Deleted {old_num_visits - visits.shape[0]} artifact visits without a visit id.")
        assert len(visits['id'].unique().tolist()) == len(visits['id'].tolist())

        visits[['id', 'duration', 'person_id']] = visits[['id', 'duration', 'person_id']].apply(pd.to_numeric)
        visits['duration'] = pd.to_timedelta(visits['duration'], unit='days')
        visits['start'] = pd.to_datetime(visits['start'], format='%Y-%m-%d')

        return visits, non_temporal, temporal

    @staticmethod
    def clean_note(note):
        # Template remove all repeated whitespaces and more than double newlines
        note = re.sub(r"[ \t]+", " ", note)
        note = re.sub("\n\n\n+", "\n\n", note)
        # Remove all leading and trailing whitespaces
        note = re.sub(r"^[ \t]+", "", note)
        note = re.sub(r"\n[ \t]+", "\n", note)
        note = re.sub(r"[ \t]$", "", note)
        note = re.sub(r"[ \t]\n", "\n", note)
        # Remove whitespaces before colon at the end of the line
        note = re.sub(r"\s*\.$", ".", note)
        note = re.sub(r"\s*\.\n", ".\n", note)
        # Remove repeated dots and the end of the line
        note = re.sub(r"\.+$", ".", note)
        note = re.sub(r"\.+\n", ".\n", note)
        # Remove whitespaces before colon at the end of the line
        note = re.sub(r"\s*\.$", ".", note)
        note = re.sub(r"\s*\.\n", ".\n", note)
        # Template remove all repeated whitespaces and more than double newlines
        note = re.sub(r"[ \t]+", " ", note)
        note = re.sub("\n\n\n+", "\n\n", note)
        # Remove repetitive whitespace colon sequences
        # Ignore for ... in creditg dataset
        if '... ' not in note:
            note = re.sub(r"(\s*\.)+ +", ". ", note)

        return note


def lower_first_char_if_second_low(s):
    return (s if s[1].isupper() else s[0].lower() + s[1:]) if (not pd.isna(s) and len(s) > 1) else s


def read_shortened_concepts(file_name):
    # Read dictionaries for shortening concepts
    if file_name is None:
        return {}
    with open('/root/omop-pkg/misc/shorten_concepts/' + file_name) as f:
        shortened_concepts = eval(f.readline())
        for k, v in shortened_concepts.items():
            v = v.strip()
            v = v if v[-1].isalnum() else v[:-1]
            # Lower first character if second not upper case
            shortened_concepts[k] = lower_first_char_if_second_low(v)
    # Lower all values longer than four character that are all upper case
    return {k.lower(): v.lower() if (len(v) > 4 and v.isupper()) else v for k, v in shortened_concepts.items()}

