import copy


def preprocess(example):
    def _add_adjusted_col_offsets(table):
        """Add adjusted column offsets to take into account multi-column cells."""
        adjusted_table = []
        for row in table:
            real_col_index = 0
            adjusted_row = []
            for cell in row:
                adjusted_cell = copy.deepcopy(cell)
                adjusted_cell["adjusted_col_start"] = real_col_index
                adjusted_cell["adjusted_col_end"] = (
                        adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"])
                real_col_index += adjusted_cell["column_span"]
                adjusted_row.append(adjusted_cell)
            adjusted_table.append(adjusted_row)
        return adjusted_table

    def _get_heuristic_row_headers(adjusted_table, row_index, col_index):
        """Heuristic to find row headers."""
        row_headers = []
        row = adjusted_table[row_index]
        for i in range(0, col_index):
            if row[i]["is_header"]:
                row_headers.append(row[i])
        return row_headers

    def _get_heuristic_col_headers(adjusted_table, row_index, col_index):
        """Heuristic to find column headers."""
        adjusted_cell = adjusted_table[row_index][col_index]
        adjusted_col_start = adjusted_cell["adjusted_col_start"]
        adjusted_col_end = adjusted_cell["adjusted_col_end"]
        col_headers = []
        for r in range(0, row_index):
            row = adjusted_table[r]
            for cell in row:
                if (cell["adjusted_col_start"] < adjusted_col_end and
                        cell["adjusted_col_end"] > adjusted_col_start):
                    if cell["is_header"]:
                        col_headers.append(cell)

        return col_headers

    table = example['table']
    cell_indices = example["highlighted_cells"]
    table_str = ""
    if example['table_page_title']:
        table_str += "<page_title> " + example['table_page_title'] + " </page_title> "
    if example['table_section_title']:
        table_str += "<section_title> " + example['table_section_title'] + " </section_title> "

    table_str += "<table> "
    adjusted_table = _add_adjusted_col_offsets(table)
    for r_index, row in enumerate(table):
        row_str = "<row> "
        for c_index, col in enumerate(row):

            row_headers = _get_heuristic_row_headers(adjusted_table, r_index, c_index)
            col_headers = _get_heuristic_col_headers(adjusted_table, r_index, c_index)

            # Distinguish between highlighted and non-highlighted cells.
            if [r_index, c_index] in cell_indices:
                start_cell_marker = "<highlighted_cell> "
                end_cell_marker = "</highlighted_cell> "
            else:
                start_cell_marker = "<c> "
                end_cell_marker = "</c> "

            # The value of the cell.
            item_str = start_cell_marker + col["value"] + " "

            # All the column headers associated with this cell.
            for col_header in col_headers:
                item_str += "<col_header> " + col_header["value"] + " </col_header> "

            # All the row headers associated with this cell.
            for row_header in row_headers:
                item_str += "<row_header> " + row_header["value"] + " </row_header> "

            item_str += end_cell_marker
            row_str += item_str

        row_str += "</row> "
        table_str += row_str

    table_str += "</table>"

    example['linearized_table'] = '<s>' + table_str + '\n' + '\n'
    return example
