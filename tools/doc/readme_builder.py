from pandas import DataFrame, read_csv
import pandas as pd
import os

# FIXME: arg
opentraj_root = "/home/cyrus/workspace2/OpenTraj"
output_str_ = ''


def print_to_readme(txt, with_newline=True):
    global output_str_
    output_str_ += txt
    if with_newline:
        output_str_ += '\n'


def build_table(headers, items, skip_columns=[]):
    num_rows = len(items)
    header_inds = [i for i in range(len(headers)) if i not in skip_columns]
    num_cols = len(header_inds)

    header_str = '| '
    seperator_str = '|'
    for header_ind in header_inds:
        header_str += headers[header_ind] + ' | '
        seperator_str += '----|'

    content_str = ''

    for ii in range(num_rows):
        content_str += '| '
        for jj in header_inds:
            # FIXME: remove it from here
            item_ij = str(items[ii][jj]) if str(items[ii][jj]) != 'nan' else ' '
            item_ij = item_ij.replace('|', '')  # `|` characters cause mess!
            content_str += item_ij + ' | '
        content_str += '\n'

    return header_str + '\n' + seperator_str + '\n' + content_str


def build_readme():
    input_readme = os.path.join(opentraj_root, "README.md")

    input_str = ''
    if os.path.exists(input_readme):
        with open(input_readme, 'r') as input_readme_file:
            input_str = input_readme_file.read()
    # output_str = ''

    # Todo: find beginning and end of table 1
    begin_str_table_main = '<!--begin(table_main)-->'
    end_str_table_main = '<!--end(table_main)-->'
    begin_ind_table_main = input_str.rfind(begin_str_table_main)
    end_ind_table_main = input_str.rfind(end_str_table_main) + len(end_str_table_main)

    # FIXME: Build Table 1
    excel_file_table_main = os.path.join(opentraj_root, "tools/doc/public-datasets.xls")
    df_main = pd.read_excel(excel_file_table_main, header=[0])

    # df_main.Description
    headers_table_main = df_main.axes[1].values[:4].tolist()
    items_table_main = df_main.values[:, :4].tolist()

    for ii, row_i in enumerate(items_table_main):
        ntraj = df_main.nTraj.values.tolist()[ii]
        coord = df_main.Coord.values.tolist()[ii]
        fps = df_main.FPS.values.tolist()[ii]
        density = df_main.Density.values.tolist()[ii]

        if str(ntraj) == 'nan' or str(coord) == 'nan' or str(fps) == 'nan' or str(density)== 'nan':
            continue

        items_table_main[ii][2] += " `#Traj:[%s]` `Coord=%s` `FPS=%s` `Density=%s` " \
                                   %(ntraj, coord, fps, density)

    # selected_header_inds = [0, 1, 6, 7, 8]  # ['Sample', 'Name', 'Description', 'REF']
    # skip_columns = list(range(len(headers_table_main)))
    # skip_columns.pop(selected_header_inds)
    skip_columns = []


    print_to_readme(input_str[:begin_ind_table_main])
    print_to_readme(begin_str_table_main)
    print_to_readme(build_table(headers_table_main, items_table_main, skip_columns))
    print_to_readme(end_str_table_main)

    cursor = end_ind_table_main

    # ETH Dataset
    excel_file_table_benchmarks = os.path.join(opentraj_root, "tools/doc/opentraj-benchmarks.xls")
    # Todo: one option is to upload the doc into dropbox and download it each time
    # wget -P opentraj_root/tools/doc/ -N https://www.dropbox.com/s/fwkyrqv1bcnwye8/OpenTraj-Benchmark-Datasets.xlsx

    df_ETH = pd.read_excel(excel_file_table_benchmarks, header=[0], sheet_name='ETH')

    headers_table_ETH = df_ETH.axes[1].values.tolist()
    items_table_ETH = df_ETH.values.tolist()

    begin_str_table_ETH = '<!--begin(table_ETH)-->'
    end_str_table_ETH = '<!--end(table_ETH)-->'
    begin_ind_table_ETH = input_str.rfind(begin_str_table_ETH)
    end_ind_table_ETH = input_str.rfind(end_str_table_ETH) + len(end_str_table_ETH)

    # Add texts before table - ETH
    print_to_readme(input_str[cursor:begin_ind_table_ETH])
    # return
    print_to_readme(begin_str_table_ETH)
    print_to_readme(build_table(headers_table_ETH, items_table_ETH))
    print_to_readme(end_str_table_ETH)
    cursor = end_ind_table_ETH



    # Append rest of the text
    print_to_readme(input_str[cursor:])


if __name__ == "__main__":
    build_readme()
    print(output_str_)

    output_readme = os.path.join(opentraj_root, "README_new.md")
    with open(output_readme, 'w') as out_file:
        out_file.write(output_str_)