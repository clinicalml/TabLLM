from collections import OrderedDict

import pandas as pd
import re


########################################################################################################################
# creditg
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
creditg_feature_names = [
    ('checking_status', 'Status of existing checking account'),
    ('duration', 'Duration in month'),
    ('credit_history', 'Credit history '),
    ('purpose', 'Purpose'),
    ('credit_amount', 'Credit amount'),
    ('savings_status', 'Savings account/bonds'),
    ('employment', 'Present employment since'),
    ('installment_commitment', 'Installment rate in percentage of disposable income'),
    ('personal_status', 'Personal status and sex'),
    ('other_parties', 'Other debtors / guarantors'),
    ('residence_since', 'Present residence since'),
    ('property_magnitude', 'Property'),
    ('age', 'Age in years'),
    ('other_payment_plans', 'Other installment plans'),
    ('housing', 'Housing'),
    ('existing_credits', 'Number of existing credits at this bank'),
    ('job', 'Job'),
    ('num_dependents', 'Number of people being liable to provide maintenance for'),
    ('own_telephone', 'Telephone'),
    ('foreign_worker', 'foreign worker')
]
checking_status_dict = {'<0': '< 0 DM', '0<=X<200': '0 <= ... < 200 DM', '>=200': '>= 200 DM', 'no checking': 'no checking account'}
credit_history_dict = {'no credits/all paid': 'no credits taken/ all credits paid back duly', 'all paid': 'all credits at this bank paid back duly', 'existing paid': 'existing credits paid back duly till now', 'delayed previously': 'delay in paying off in the past', 'critical/other existing credit': 'critical account/ other credits existing (not at this bank)'}
purpose_dict = {'new car': 'car (new)', 'used car': 'car (used)', 'furniture/equipment': 'furniture/equipment', 'radio/tv': 'radio/television', 'domestic appliance': 'domestic appliances', 'repairs': 'repairs', 'education': 'education', 'retraining': 'retraining', 'business': 'business', 'other': 'others'}
savings_status_dict = {'<100': '... < 100 DM', '100<=X<500': '100 <= ... < 500 DM', '500<=X<1000': '500 <= ... < 1000 DM', '>=1000': '... >= 1000 DM', 'no known savings': 'unknown/ no savings account'}
employment_dict = {'unemployed': 'unemployed', '<1': '... < 1 year', '1<=X<4': '1 <= ... < 4 years', '4<=X<7': '4 <= ... < 7 years', '>=7': '... >= 7 years',}
personal_status_dict = {'female div/dep/mar': 'female : divorced/separated/married', 'male div/sep': 'male : divorced/separated', 'male mar/wid': 'male : married/widowed', 'male single': 'male : single'}
other_parties_dict = {'none': 'none', 'co applicant': 'co-applicant', 'guarantor': 'guarantor'}
property_magnitude_dict = {'car': 'car or other, not in attribute 6', 'life insurance': 'building society savings agreement/ life insurance', 'no known property': 'unknown / no property', 'real estate': 'real estate'}
job_dict = {'high qualif/self emp/mgmt': 'management/ self-employed/ highly qualified employee/ officer', 'skilled': 'skilled employee / official', 'unemp/unskilled non res': 'unemployed/ unskilled - non-resident', 'unskilled resident': 'unskilled - resident'}
own_telephone_dict = {'none': 'none', 'yes': 'yes, registered under the customers name'}
template_config_creditg = {
    'pre': {
        'checking_status': lambda x: checking_status_dict[x],
        'duration': lambda x: f"{int(x)}",
        'credit_history': lambda x: credit_history_dict[x],
        'purpose': lambda x: purpose_dict[x],
        'credit_amount': lambda x: f"{int(x)}",
        'savings_status': lambda x: savings_status_dict[x],
        'employment_status': lambda x: employment_dict[x],
        'installment_commitment': lambda x: f"{int(x)}",
        'personal_status': lambda x: personal_status_dict[x],
        'other_parties': lambda x: other_parties_dict[x],
        'residence_since': lambda x: f"{int(x)}",
        'property_magnitude': lambda x: property_magnitude_dict[x],
        'age': lambda x: f"{int(x)}",
        'existing_credits': lambda x: f"{int(x)}",
        'job': lambda x: job_dict[x],
        'own_telephone': lambda x: own_telephone_dict[x]
    }
}
template_creditg = ' '.join(['The ' + v + ' is ${' + k + '}.' for k, v in creditg_feature_names])
template_creditg_list = '\n'.join(['- ' + v + ': ${' + k + '}' for k, v in creditg_feature_names])
template_creditg_list_values = '\n'.join(['${' + k + '}' for k, v in creditg_feature_names])
creditg_permutation = [17, 19, 20, 15, 11, 7, 3, 10, 6, 13, 12, 2, 8, 4, 14, 1, 16, 9, 18, 5]
template_creditg_list_permuted = '\n'.join(['- ' + x[1] + ': ${' + creditg_feature_names[creditg_permutation[i] - 1][0] + '}' for i, x in enumerate(creditg_feature_names)])
template_config_creditg_list = template_config_creditg
template_config_creditg_list_permuted = template_config_creditg
template_config_creditg_list_values = template_config_creditg_list
template_creditg_list_shuffled = template_creditg_list
template_config_creditg_list_shuffled = template_config_creditg_list


########################################################################################################################
# blood
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
blood_feature_names = [
    ('recency', 'Recency - months since last donation'),
    ('frequency', 'Frequency - total number of donation'),
    ('monetary', 'Monetary - total blood donated in c.c.'),
    ('time', 'Time - months since first donation'),
]
template_config_blood = {
    'pre': {
        'recency': lambda x: f"{int(x)}",
        'frequency': lambda x: f"{int(x)}",
        'monetary': lambda x: f"{int(x)}",
        'time': lambda x: f"{int(x)}",
    }
}
template_blood = ' '.join(['The ' + v + ' is ${' + k + '}.' for k, v in blood_feature_names])
template_blood_list = '\n'.join(['- ' + v + ': ${' + k + '}' for k, v in blood_feature_names])
template_blood_list_values = '\n'.join(['${' + k + '}' for k, v in blood_feature_names])
blood_permutation = [4, 3, 1, 2]
template_blood_list_permuted = '\n'.join(['- ' + x[1] + ': ${' + blood_feature_names[blood_permutation[i] - 1][0] + '}' for i, x in enumerate(blood_feature_names)])
template_config_blood_list = template_config_blood
template_config_blood_list_permuted = template_config_blood
template_config_blood_list_values = template_config_blood_list
template_blood_list_shuffled = template_blood_list
template_config_blood_list_shuffled = template_config_blood_list

########################################################################################################################
# bank
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/bank+marketing
# and https://www.openml.org/search?type=data&sort=runs&id=1461&status=active
bank_feature_names = [
    ('age', 'age'),
    ('job', 'type of job'),
    ('marital', 'marital status'),
    ('education', 'education'),
    ('default', 'has credit in default?'),
    ('balance', 'average yearly balance, in euros'),
    ('housing', 'has housing loan?'),
    ('loan', 'has personal loan?'),
    ('contact', 'contact communication type'),
    ('day', 'last contact day of the month'),
    ('month', 'last contact month of year'),
    ('duration', 'last contact duration, in seconds'),
    ('campagin', 'number of contacts performed during this campaign and for this client'),
    ('pdays', 'number of days that passed by after the client was last contacted from a previous campaign'),
    ('previous', 'number of contacts performed before this campaign and for this client'),
    ('poutcome', 'outcome of the previous marketing campaign'),
]
template_config_bank = {
    'pre': {
        'age': lambda x: f"{int(x)}",
        'balance': lambda x: f"{int(x)}",
        'day': lambda x: f"{int(x)}",
        'duration': lambda x: f"{int(x)}",
        'campaign': lambda x: f"{int(x)}",
        'pdays': lambda x: f"{int(x)}" if x != -1 else 'client was not previously contacted',
        'previous': lambda x: f"{int(x)}",
    }
}
template_bank = ' '.join(['The ' + v + ' is ${' + k + '}.' for k, v in bank_feature_names])
template_bank_list = '\n'.join(['- ' + v + ': ${' + k + '}' for k, v in bank_feature_names])
template_bank_list_values = '\n'.join(['${' + k + '}' for k, v in bank_feature_names])
bank_permutation = [14, 11, 4, 15, 8, 16, 12, 1, 6, 5, 13, 7, 9, 3, 10, 2]
template_bank_list_permuted = '\n'.join(['- ' + x[1] + ': ${' + bank_feature_names[bank_permutation[i] - 1][0] + '}' for i, x in enumerate(bank_feature_names)])
template_config_bank_list = template_config_bank
template_config_bank_list_permuted = template_config_bank
template_config_bank_list_values = template_config_bank_list
template_bank_list_shuffled = template_bank_list
template_config_bank_list_shuffled = template_config_bank_list


########################################################################################################################
# jungle
########################################################################################################################
# Use description from: https://arxiv.org/abs/1604.07312
jungle_feature_names = [
    ('white_piece0_strength', 'white piece strength'),
    ('white_piece0_file', 'white piece file'),
    ('white_piece0_rank', 'white piece rank'),
    ('black_piece0_strength', 'black piece strength'),
    ('black_piece0_file', 'black piece file'),
    ('black_piece0_rank', 'black piece rank')
]
template_config_jungle = {
    'pre': {
        'white_piece0_strength': lambda x: f"{int(x)}",
        'white_piece0_file': lambda x: f"{int(x)}",
        'white_piece0_rank': lambda x: f"{int(x)}",
        'black_piece0_strength': lambda x: f"{int(x)}",
        'black_piece0_file': lambda x: f"{int(x)}",
        'black_piece0_rank': lambda x: f"{int(x)}",
    }
}
template_jungle = ' '.join(['The ' + v + ' is ${' + k + '}.' for k, v in jungle_feature_names])
template_jungle_list = '\n'.join(['- ' + v + ': ${' + k + '}' for k, v in jungle_feature_names])
template_jungle_list_values = '\n'.join(['${' + k + '}' for k, v in jungle_feature_names])
jungle_permutation = [4, 5, 2, 1, 6, 3]
template_jungle_list_permuted = '\n'.join(['- ' + x[1] + ': ${' + jungle_feature_names[jungle_permutation[i] - 1][0] + '}' for i, x in enumerate(jungle_feature_names)])
template_config_jungle_list = template_config_jungle
template_config_jungle_list_permuted = template_config_jungle
template_config_jungle_list_values = template_config_jungle_list
template_jungle_list_shuffled = template_jungle_list
template_config_jungle_list_shuffled = template_config_jungle_list


########################################################################################################################
# calhousing
########################################################################################################################
# Use description from: Pace and Barry (1997), "Sparse Spatial Autoregressions", Statistics and Probability Letters.
calhousing_feature_names = [
    ('median_income', 'median income'),
    ('housing_median_age', 'median age'),
    ('total_rooms', 'total rooms'),
    ('total_bedrooms', 'total bedrooms'),
    ('population', 'population'),
    ('households', 'households'),
    ('latitude', 'latitude'),
    ('longitude', 'longitude'),
]
template_config_calhousing = {
    'pre': {
        'median_income': lambda x: f"{x:.4f}",
        'housing_median_age': lambda x: f"{int(x)}",
        'total_rooms': lambda x: f"{int(x)}",
        'total_bedrooms': lambda x: f"{int(x)}",
        'population': lambda x: f"{int(x)}",
        'households': lambda x: f"{int(x)}",
        'latitude': lambda x: f"{x:.2f}",
        'longitude': lambda x: f"{x:.2f}",
    }
}
template_calhousing = ' '.join(['The ' + v + ' is ${' + k + '}.' for k, v in calhousing_feature_names])
template_calhousing_list = '\n'.join(['- ' + v + ': ${' + k + '}' for k, v in calhousing_feature_names])
template_calhousing_list_values = '\n'.join(['${' + k + '}' for k, v in calhousing_feature_names])
calhousing_permutation = [4, 5, 8, 3, 2, 7, 1, 6]
template_calhousing_list_permuted = '\n'.join(['- ' + x[1] + ': ${' + calhousing_feature_names[calhousing_permutation[i] - 1][0] + '}' for i, x in enumerate(calhousing_feature_names)])
template_config_calhousing_list = template_config_calhousing
template_config_calhousing_list_permuted = template_config_calhousing
template_config_calhousing_list_values = template_config_calhousing_list
template_calhousing_list_shuffled = template_calhousing_list
template_config_calhousing_list_shuffled = template_config_calhousing_list

########################################################################################################################
# car
########################################################################################################################
prices_dict = {'vhigh': 'very high', 'high': 'high', 'med': 'medium', 'low': 'low'}
doors_dict = {'2': 'two', '3': 'three', '4': 'four', '5more': 'five or more'}
persons_dict = {'2': 'two', '4': 'four', 'more': 'more than four'}
lug_boot_dict = {'big': 'big', 'med': 'medium', 'small': 'small'}
safety_dict = {'high': 'high', 'med': 'medium', 'low': 'low'}
template_config_car = {
    'pre': {
        'doors': lambda x: doors_dict[x],
        'persons': lambda x: persons_dict[x],
        'lug_boot': lambda x: lug_boot_dict[x],
        'safety_dict': lambda x: safety_dict[x],
        'buying': lambda x: prices_dict[x],
        'maint': lambda x: prices_dict[x]
    }
}
template_car = 'The Buying price is ${buying}. ' \
               'The Doors is ${doors}. ' \
               'The Maintenance costs is ${maint}. ' \
               'The Persons is ${persons}. ' \
               'The Safety score is ${safety_dict}. ' \
               'The Trunk size is ${lug_boot}.'
template_car_list = '- Buying price: ${buying}\n' \
                    '- Doors: ${doors}\n' \
                    '- Maintenance costs: ${maint}\n' \
                    '- Persons: ${persons}\n' \
                    '- Safety score: ${safety_dict}\n' \
                    '- Trunk size: ${lug_boot}'
template_config_car_list = template_config_car
template_car_list_permuted = '- Buying price: ${safety_dict}\n' \
                             '- Doors: ${buying}\n' \
                             '- Maintenance costs: ${lug_boot}\n' \
                             '- Persons: ${doors}\n' \
                             '- Safety score: ${maint}\n' \
                             '- Trunk size: ${persons}'
template_config_car_list_permuted = template_config_car
template_car_list_values = '${buying}\n' \
                           '${doors}\n' \
                           '${maint}\n' \
                           '${persons}\n' \
                           '${safety_dict}\n' \
                           '${lug_boot}'
template_config_car_list_values = template_config_car_list
template_car_list_shuffled = template_car_list
template_config_car_list_shuffled = template_config_car_list


########################################################################################################################
# income
########################################################################################################################
gender_categories = ['female', 'male']
race_categories = ['race not recorded', 'hispanic or latino', 'asian', 'black or african american',
                   'american indian or alaska native', 'white', 'native hawaiian or other pacific islander']
female_name = 'Mary Smith'
male_name = 'James Smith'
female_pronoun = 'she'
male_pronoun = 'he'

# Compiled from https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/1990-census-sic-codes.pdf and
# https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2002-census-occupation-codes.xls
occupation_dict = {
    'Tech-support': 'in the technology and support sector',
    'Craft-repair': 'in the craft and repair sector',
    'Other-service': 'in the service sector',
    'Sales': 'in the sales sector',
    'Exec-managerial': 'in execution and management',
    'Prof-specialty': 'in a professional specialty',
    'Handlers-cleaners': 'in the cleaning and maintenance sector',
    'Machine-op-inspct': 'as a machine operator and inspector',
    'Adm-clerical': 'in office and administrative support',
    'Farming-fishing': 'in the agriculture, forestry, and fisheries sector',
    'Transport-moving': 'in the transportation, communication, and other public utilities sector',
    'Priv-house-serv': 'in their private household',
    'Protective-serv': 'in the protective services sector',
    'Armed-Forces': 'in the armed forces'
}
occupation_dict_list = {
    'Tech-support': 'technology and support sector',
    'Craft-repair': 'craft and repair sector',
    'Other-service': 'service sector',
    'Sales': 'sales sector',
    'Exec-managerial': 'execution and management',
    'Prof-specialty': 'professional specialty',
    'Handlers-cleaners': 'cleaning and maintenance sector',
    'Machine-op-inspct': 'machine operator and inspector',
    'Adm-clerical': 'office and administrative support',
    'Farming-fishing': 'agriculture, forestry, and fisheries sector',
    'Transport-moving': 'transportation, communication, and other public utilities sector',
    'Priv-house-serv': 'private household',
    'Protective-serv': 'protective services sector',
    'Armed-Forces': 'armed forces'
}
workclass_dict = {
    'Private': 'as a private sector employee',
    'Local-gov': 'for the local government',
    'State-gov': 'for the state government',
    'Federal-gov': 'for the federal government',
    'Self-emp-not-inc': 'as an owner of a non-incorporated business, professional practice, or farm',
    'Self-emp-inc': 'as a an owner of a incorporated business, professional practice, or farm',
    'Without-pay': 'without pay in a for-profit family business or farm',
    'Never-worked': 'never worked',
}
workclass_dict_list = {
    'Private': 'private sector employee',
    'Local-gov': 'local government',
    'State-gov': 'state government',
    'Federal-gov': 'federal government',
    'Self-emp-not-inc': 'owner of a non-incorporated business, professional practice, or farm',
    'Self-emp-inc': 'owner of a incorporated business, professional practice, or farm',
    'Without-pay': 'without pay in a for-profit family business or farm',
    'Never-worked': 'never worked',
}
# From: https://www.census.gov/content/dam/Census/library/publications/2007/dec/10_education.pdf
education_dict = {
    'Doctorate': 'has a doctoral degree',
    'Prof-school': 'has a professional degree',
    'Masters': 'has a master\'s degree',
    'Bachelors': 'has a bachelor\'s degree',
    'Assoc-acdm': 'has an associate\'s degree',
    'Assoc-voc': 'went to college for one or more years without a degree',
    'Some-college': 'went to college for less than one year',
    'HS-grad': 'is a high school graduate',
    '12th': 'finished 12th class without diploma',
    '11th': 'finished 11th class',
    '10th': 'finished 10th class',
    '9th': 'finished 9th class',
    '7th-8th': 'finished 8th class',
    '5th-6th': 'finished 6th class',
    '1st-4th': 'finished 4th class',
    'Preschool': 'completed no schooling'
}
education_dict_list = {
    'Doctorate': 'doctoral degree',
    'Prof-school': 'professional degree',
    'Masters': 'master\'s degree',
    'Bachelors': 'bachelor\'s degree',
    'Assoc-acdm': 'associate\'s degree',
    'Assoc-voc': 'college for one or more years without a degree',
    'Some-college': 'college for less than one year',
    'HS-grad': 'high school graduate',
    '12th': 'finished 12th class without diploma',
    '11th': 'finished 11th class',
    '10th': 'finished 10th class',
    '9th': 'finished 9th class',
    '7th-8th': 'finished 8th class',
    '5th-6th': 'finished 6th class',
    '1st-4th': 'finished 4th class',
    'Preschool': 'no schooling'
}
# From https://www.census.gov/programs-surveys/cps/technical-documentation/subject-definitions.html#householder
relationship_dict = {
    'Wife': 'and is the wife of the head of the household',
    'Own-child': 'and is a child of the head of the household',
    'Husband': 'and is the husband of the head of the household',
    'Not-in-family': 'and is not in a family',
    'Other-relative': 'and is an other relative of the head of the household',
    'Unmarried': 'and is not married to the head of the household'
}
relationship_dict_list = {
    'Wife': 'wife',
    'Own-child': 'own child',
    'Husband': 'husband',
    'Not-in-family': 'not in a family',
    'Other-relative': 'other relative',
    'Unmarried': 'unmarried'
}
template_config_income = {
    'pre': {
        'race': lambda r: None if r.lower() == 'other' else r,
        'marital_status': lambda ms: 'married' if ms.lower().startswith('married-') else
        ('never married' if ms.lower() == 'never-married' else ms.lower()),
        'native_country': lambda nc: 'United States' if nc in ['United-States', 'Outlying-US(Guam-USVI-etc)']
        else (None if pd.isna(nc) else ('South Korea' if nc.lower() == 'South' else nc)),
        'occupation': lambda o: occupation_dict_list.get(o, ''),
        'workclass': lambda w: workclass_dict_list.get(w, ''),
        'education': lambda e: education_dict_list.get(e)
    },
}
template_income = 'The Age is ${age}. ' \
                  'The Race is ${race}. ' \
                  'The Sex is ${sex}. ' \
                  'The Marital status is ${marital_status}. ' \
                  'The Relation to head of the household is ${relationship}. ' \
                  'The Native country is ${native_country}. ' \
                  'The Occupation is ${occupation}. ' \
                  'The Work class is ${workclass}. ' \
                  'The Capital gain last year is ${capital_gain}. ' \
                  'The Capital loss last year is ${capital_loss}. ' \
                  'The Education is ${education}. ' \
                  'The Work hours per week is ${hours_per_week}.'
template_config_income_list = template_config_income
template_income_list = '- Age: ${age}\n' \
                       '- Race: ${race}\n' \
                       '- Sex: ${sex}\n' \
                       '- Marital status: ${marital_status}\n' \
                       '- Relation to head of the household: ${relationship}\n' \
                       '- Native country: ${native_country}\n' \
                       '- Occupation: ${occupation}\n' \
                       '- Work class: ${workclass}\n' \
                       '- Capital gain last year: ${capital_gain}\n' \
                       '- Capital loss last year: ${capital_loss}\n' \
                       '- Education: ${education}\n' \
                       '- Work hours per week: ${hours_per_week}'
template_income_list_permuted = '- Age: ${workclass}\n' \
                                '- Race: ${occupation}\n' \
                                '- Sex: ${hours_per_week}\n' \
                                '- Marital status: ${age}\n' \
                                '- Relation to head of the household: ${capital_gain}\n' \
                                '- Native country: ${race}\n' \
                                '- Occupation: ${relationship}\n' \
                                '- Work class: ${education}\n' \
                                '- Capital gain last year: ${marital_status}\n' \
                                '- Capital loss last year: ${native_country}\n' \
                                '- Education: ${sex}\n' \
                                '- Work hours per week: ${capital_loss}'
template_config_income_list_permuted = template_config_income_list
template_income_list_values = '${age}\n' \
                              '${race}\n' \
                              '${sex}\n' \
                              '${marital_status}\n' \
                              '${relationship}\n' \
                              '${native_country}\n' \
                              '${occupation}\n' \
                              '${workclass}\n' \
                              '${capital_gain}\n' \
                              '${capital_loss}\n' \
                              '${education}\n' \
                              '${hours_per_week}'
template_config_income_list_values = template_config_income_list
template_income_list_shuffled = template_income_list
template_config_income_list_shuffled = template_config_income_list
template_income_list_importance = template_income_list
template_config_income_list_importance = template_config_income_list

########################################################################################################################
# heart
########################################################################################################################
# Used descriptions from: https://www.kaggle.com/code/azizozmen/heart-failure-predict-8-classification-techniques
chest_paint_types_list = {'TA': 'typical angina', 'ATA': 'atypical angina', 'NAP': 'non-anginal pain', 'ASY': 'asymptomatic'}
rest_ecg_results = {
    'Normal': 'normal',
    'ST': 'ST-T wave abnormality',
    'LVH': 'probable or definite left ventricular hypertrophy'
}
st_slopes = {'Up': 'upsloping', 'Flat': 'flat', 'Down': 'downsloping'}
template_config_heart = {
    'pre': {
        'Sex': lambda x: 'male' if x == 'M' else 'female',
        'ChestPainType': lambda x: chest_paint_types_list[x],
        'FastingBS': lambda x: 'yes' if x == 1 else 'no',
        'ExerciseAngina': lambda x: 'yes' if x == 'Y' else 'no',
        'ST_Slope': lambda x: st_slopes[x],
        'RestingECG': lambda x: rest_ecg_results[x]
    }
}
template_heart = 'The Age of the patient is ${Age}. ' \
                 'The Sex of the patient is ${Sex}. ' \
                 'The Chest pain type is ${ChestPainType}. ' \
                 'The Resting blood pressure is ${RestingBP}. ' \
                 'The Serum cholesterol is ${Cholesterol}. ' \
                 'The Fasting blood sugar > 120 mg/dl is ${FastingBS}. ' \
                 'The Resting electrocardiogram results is ${RestingECG}. ' \
                 'The Maximum heart rate achieved is ${MaxHR}. ' \
                 'The Exercise-induced angina is ${ExerciseAngina}. ' \
                 'The ST depression induced by exercise relative to rest is ${Oldpeak}. ' \
                 'The Slope of the peak exercise ST segment is ${ST_Slope}.'
template_config_heart_list = template_config_heart
template_heart_list = '- Age of the patient: ${Age}\n' \
                      '- Sex of the patient: ${Sex}\n' \
                      '- Chest pain type: ${ChestPainType}\n' \
                      '- Resting blood pressure: ${RestingBP}\n' \
                      '- Serum cholesterol: ${Cholesterol}\n' \
                      '- Fasting blood sugar > 120 mg/dl: ${FastingBS}\n' \
                      '- Resting electrocardiogram results: ${RestingECG}\n' \
                      '- Maximum heart rate achieved: ${MaxHR}\n' \
                      '- Exercise-induced angina: ${ExerciseAngina}\n' \
                      '- ST depression induced by exercise relative to rest: ${Oldpeak}\n' \
                      '- Slope of the peak exercise ST segment: ${ST_Slope}'
template_heart_list_permuted = '- Age of the patient: ${RestingECG}\n' \
                               '- Sex of the patient: ${Age}\n' \
                               '- Chest pain type: ${Cholesterol}\n' \
                               '- Resting blood pressure: ${ST_Slope}\n' \
                               '- Serum cholesterol: ${RestingBP}\n' \
                               '- Fasting blood sugar > 120 mg/dl: ${ChestPainType}\n' \
                               '- Resting electrocardiogram results: ${ExerciseAngina}\n' \
                               '- Maximum heart rate achieved: ${Sex}\n' \
                               '- Exercise-induced angina: ${Oldpeak}\n' \
                               '- ST depression induced by exercise relative to rest: ${MaxHR}\n' \
                               '- Slope of the peak exercise ST segment: ${FastingBS}'
template_config_heart_list_permuted = template_config_heart_list
template_heart_list_values = '${Age}\n' \
                             '${Sex}\n' \
                             '${ChestPainType}\n' \
                             '${RestingBP}\n' \
                             '${Cholesterol}\n' \
                             '${FastingBS}\n' \
                             '${RestingECG}\n' \
                             '${MaxHR}\n' \
                             '${ExerciseAngina}\n' \
                             '${Oldpeak}\n' \
                             '${ST_Slope}'
template_config_heart_list_values = template_config_heart_list
template_heart_list_shuffled = template_heart_list
template_config_heart_list_shuffled = template_config_heart_list

########################################################################################################################
# diabetes
########################################################################################################################
# Used descriptions from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf
template_config_diabetes = {
    'pre': {
        'Age': lambda x: f"{int(x)}",
        'Pregnancies': lambda x: f"{int(x)}",
        'BloodPressure': lambda x: f"{int(x)}",
        'SkinThickness': lambda x: f"{int(x)}",
        'Glucose': lambda x: f"{int(x)}",
        'Insulin': lambda x: f"{int(x)}",
        'BMI': lambda x: f"{x:.1f}",
        'DiabetesPedigreeFunction': lambda x: f"{x:.3f}"
    }
}
template_diabetes = 'The Age is ${Age}. ' \
                    'The Number of times pregnant is ${Pregnancies}. ' \
                    'The Diastolic blood pressure is ${BloodPressure}. ' \
                    'The Triceps skin fold thickness is ${SkinThickness}. ' \
                    'The Plasma glucose concentration at 2 hours in an oral glucose tolerance test (GTT) is ' \
                    '${Glucose}. ' \
                    'The 2-hour serum insulin is ${Insulin}. ' \
                    'The Body mass index is ${BMI}. ' \
                    'The Diabetes pedigree function is ${DiabetesPedigreeFunction}.'
template_config_diabetes_list = template_config_diabetes
template_diabetes_list = '- Age: ${Age}\n' \
                         '- Number of times pregnant: ${Pregnancies}\n' \
                         '- Diastolic blood pressure: ${BloodPressure}\n' \
                         '- Triceps skin fold thickness: ${SkinThickness}\n' \
                         '- Plasma glucose concentration at 2 hours in an oral glucose tolerance test (GTT): ' \
                         '${Glucose}\n' \
                         '- 2-hour serum insulin: ${Insulin}\n' \
                         '- Body mass index: ${BMI}\n' \
                         '- Diabetes pedigree function: ${DiabetesPedigreeFunction}'
template_diabetes_list_permuted = '- Age: ${Glucose}\n' \
                                  '- Number of times pregnant: ${DiabetesPedigreeFunction}\n' \
                                  '- Diastolic blood pressure: ${BMI}\n' \
                                  '- Triceps skin fold thickness: ${BloodPressure}\n' \
                                  '- Plasma glucose concentration at 2 hours in an oral glucose tolerance test (GTT): ' \
                                  '${Age}\n' \
                                  '- 2-hour serum insulin: ${SkinThickness}\n' \
                                  '- Body mass index: ${Pregnancies}\n' \
                                  '- Diabetes pedigree function: ${Insulin}'
template_config_diabetes_list_permuted = template_config_diabetes_list
template_diabetes_list_values = '${Age}\n' \
                                '${Pregnancies}\n' \
                                '${BloodPressure}\n' \
                                '${SkinThickness}\n' \
                                '${Glucose}\n' \
                                '${Insulin}\n' \
                                '${BMI}\n' \
                                '${DiabetesPedigreeFunction}'
template_config_diabetes_list_values = template_config_diabetes_list
template_diabetes_list_shuffled = template_diabetes_list
template_config_diabetes_list_shuffled = template_config_diabetes_list


########################################################################################################################
# wine
########################################################################################################################
# Use data from: https://archive.ics.uci.edu/ml/datasets/wine+quality
template_config_wine = {
    'pre': {
        'fixed_acidity': lambda x: f"{x:.1f}",
        'volatile_acidity': lambda x: f"{x:.3f}",
        'citric_acid': lambda x: f"{x:.2f}",
        'residual_sugar': lambda x: f"{x:.1f}",
        'chlorides': lambda x: f"{x:.3f}",
        'free_sulfur_dioxide': lambda x: f"{int(x)}",
        'total_sulfur_dioxide': lambda x: f"{int(x)}",
        'density': lambda x: f"{x:.5f}",
        'pH': lambda x: f"{x:.2f}",
        'sulphates': lambda x: f"{x:.2f}",
        'alcohol': lambda x: f"{x:.1f}"
    }
}
template_wine = 'The fixed acidity is ${fixed_acidity}. ' \
                'The volatile acidity is ${volatile_acidity}. ' \
                'The citric acid is ${citric_acid}. ' \
                'The residual sugar is ${residual_sugar}. ' \
                'The chlorides is ${chlorides}. ' \
                'The free sulfur dioxide is ${free_sulfur_dioxide}. ' \
                'The total sulfur dioxide is ${total_sulfur_dioxide}. ' \
                'The density is ${density}. ' \
                'The pH is ${pH}. ' \
                'The sulphates is ${sulphates}. ' \
                'The alcohol is ${alcohol}.'
template_config_wine_list = template_config_wine
template_wine_list = '- fixed acidity: ${fixed_acidity}\n' \
                     '- volatile acidity: ${volatile_acidity}\n' \
                     '- citric acid: ${citric_acid}\n' \
                     '- residual sugar: ${residual_sugar}\n' \
                     '- chlorides: ${chlorides}\n' \
                     '- free sulfur dioxide: ${free_sulfur_dioxide}\n' \
                     '- total sulfur dioxide: ${total_sulfur_dioxide}\n' \
                     '- density: ${density}\n' \
                     '- pH: ${pH}\n' \
                     '- sulphates: ${sulphates}\n' \
                     '- alcohol: ${alcohol}'
template_wine_list_permuted = '- fixed acidity: ${pH}\n' \
                              '- volatile acidity: ${free_sulfur_dioxide}\n' \
                              '- citric acid: ${total_sulfur_dioxide}\n' \
                              '- residual sugar: ${citric_acid}\n' \
                              '- chlorides: ${sulphates}\n' \
                              '- free sulfur dioxide: ${volatile_acidity}\n' \
                              '- total sulfur dioxide: ${chlorides}\n' \
                              '- density: ${residual_sugar}\n' \
                              '- pH: ${fixed_acidity}\n' \
                              '- sulphates: ${alcohol}\n' \
                              '- alcohol: ${density}'
template_config_wine_list_permuted = template_config_wine_list
template_wine_list_values = '${fixed_acidity}\n' \
                            '${volatile_acidity}\n' \
                            '${citric_acid}\n' \
                            '${residual_sugar}\n' \
                            '${chlorides}\n' \
                            '${free_sulfur_dioxide}\n' \
                            '${total_sulfur_dioxide}\n' \
                            '${density}\n' \
                            '${pH}\n' \
                            '${sulphates}\n' \
                            '${alcohol}'
template_config_wine_list_values = template_config_wine_list
template_wine_list_shuffled = template_wine_list
template_config_wine_list_shuffled = template_config_wine_list
