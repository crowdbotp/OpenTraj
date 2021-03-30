from pandas import DataFrame, read_csv
import pandas as pd
import sys
import os
import re
import numpy as np


def print_to_readme(txt, with_newline=True):
    global output_str_
    output_str_ += txt
    if with_newline:
        output_str_ += '\n'


def build_table_html(headers, items, widths=[]):
    num_cols = len(headers)
    num_rows = len(items)

    out_str = '<table>\n'
    header_str = ''
    for ii, hdr_i in enumerate(headers):
        if len(widths) > ii:
            header_str += '    <th width = \'%dpx\'>%s</th>\n' % (widths[ii], hdr_i)
        else:
            header_str += '    <th>%s</th>\n' % hdr_i
    out_str += '  <tr>\n' + header_str + '  </tr>\n'

    items_str = ''
    for ii in range(num_rows):
        items_str += '  <tr>\n'
        for jj in range(num_cols):
            item_ij = str(items[ii][jj]) if str(items[ii][jj]) != 'nan' else ' '
            item_ij = item_ij.replace('|', '')  # `|` characters cause mess!
            items_str += '    <td>%s</td>\n' % str(item_ij)
        items_str += '  </tr>\n'
    out_str += items_str + '</table>\n'

    return out_str


def build_table_md(headers, items, skip_columns=[]):
    header_inds = [i for i in range(len(headers)) if i not in skip_columns]
    num_cols = len(header_inds)
    num_rows = len(items)

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
            item_ij = item_ij.replace('|', '').replace('\n', '')  # `|` characters cause mess!
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
    begin_str_table_main = '\n<!--begin(table_main)-->'
    end_str_table_main = '<!--end(table_main)-->\n'
    begin_ind_table_main = input_str.rfind(begin_str_table_main)
    end_ind_table_main = input_str.rfind(end_str_table_main) + len(end_str_table_main)

    # FIXME: Build Table 1
    df_main = pd.read_excel(excel_file_table_datasets, header=[0])

    # df_main.Description
    headers_table_main = df_main.axes[1].values[:4].tolist()
    description_header = ''.join(['&nbsp;' for i in range(50)]) +\
                         'Description' +\
                         ''.join(['&nbsp;' for i in range(50)])
    headers_table_main[2] = description_header

    items_table_main = np.array(df_main.values[:, :4].tolist())

    for ii, row_i in enumerate(items_table_main):
        img_str_begin = items_table_main[ii][0].rfind('(') + 1
        img_str_end = items_table_main[ii][0].rfind(')')
        if table_html_format:
            cols_width = [350, 100, 500, 200]
            items_table_main[ii][0] = '<img src=\'%s\' width=\'%dpx\'\>' \
                                      %(str(items_table_main[ii][0][img_str_begin:img_str_end]), cols_width[0])

        if str(items_table_main[ii][2]) == 'nan':
            items_table_main[ii][2] = ''

        ntraj = df_main.nTraj.values.tolist()[ii]
        coord = df_main.Coord.values.tolist()[ii]
        fps = df_main.FPS.values.tolist()[ii]
        density = df_main.Density.values.tolist()[ii]
        if str(ntraj) != 'nan':
            items_table_main[ii][2] += " <code>#Traj:[%s]</code>" % str(ntraj)

        if str(coord) != 'nan':
            items_table_main[ii][2] += " <code>Coord=%s</code>" % str(coord)

        if str(fps) != 'nan':
            items_table_main[ii][2] += " <code>FPS=%s</code>" % str(fps)

        if str(density) != 'nan' and not '?' in str(density):
            items_table_main[ii][2] += " <code>Density=%s</code>" % str(density)

        if table_html_format:
            refs = items_table_main[ii][3]
            items_table_main[ii][3] = ''
            ref_name = [occ.group() for occ in re.finditer('\[(.+?)\]', refs)]
            ref_link = [occ.group() for occ in re.finditer('\((.+?)\)', refs)]
            for kk in range(len(ref_link)):
                items_table_main[ii][3] += '<a href=\'%s\'>%s</a> ' % (ref_link[kk][1:-1], ref_name[kk][:])

        # print(items_table_main[ii][3])

    # selected_header_inds = [0, 1, 6, 7, 8]  # ['Sample', 'Name', 'Description', 'REF']
    # skip_columns = list(range(len(headers_table_main)))
    # skip_columns.pop(selected_header_inds)
    skip_columns = []


    print_to_readme(input_str[:begin_ind_table_main], with_newline=False)
    print_to_readme(begin_str_table_main)
    if table_html_format:
        print_to_readme(build_table_html(headers_table_main, items_table_main))
    else:
        print_to_readme(build_table_md(headers_table_main, items_table_main))
    print_to_readme(end_str_table_main, with_newline=False)

    cursor = end_ind_table_main

    # Predicttion Benchmark Table:  ETH
    # ==========================================
    # df_ETH = pd.read_excel(excel_file_table_benchmark_eth, header=[0])

    # headers_table_ETH = df_ETH.axes[1].values.tolist()
    # items_table_ETH = df_ETH.values.tolist()

    # begin_str_table_ETH = '<!--begin(table_ETH)-->'
    # end_str_table_ETH = '<!--end(table_ETH)-->'
    # begin_ind_table_ETH = input_str.rfind(begin_str_table_ETH)
    # end_ind_table_ETH = input_str.rfind(end_str_table_ETH) + len(end_str_table_ETH)
    #
    # # Add texts before table - ETH
    # print_to_readme(input_str[cursor:begin_ind_table_ETH], with_newline=False)
    #
    # print_to_readme(begin_str_table_ETH)
    # print_to_readme(build_table(headers_table_ETH, items_table_ETH))
    # print_to_readme(end_str_table_ETH, with_newline=False)
    # cursor = end_ind_table_ETH
    # ========================================

    # Append rest of the text
    print_to_readme(input_str[cursor:], with_newline=False)


if __name__ == "__main__":
    if len(sys.argv) > 2 and not sys.argv[1].startswith('--'):
        opentraj_root = sys.argv[1]
    else:
        opentraj_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # double check: to see if this script can be found where it should be
    if not os.path.exists(os.path.join(opentraj_root, 'doc/readme_builder.py')):
        print('Error! could not find true opentraj path!')
        exit(-1)

    excel_file_table_datasets = os.path.join(opentraj_root, "doc/data/opentraj-datasets.xlsx")
    excel_file_table_benchmark_eth = os.path.join(opentraj_root, "doc/data/opentraj-benchmark-eth.xlsx")

    if '--download-tables' in sys.argv:
        # Todo: one option is to upload the doc into dropbox and download it each time
        os.system(f'wget -q -O {excel_file_table_datasets}'
                  f' -N "https://ethercalc.org/5xdmtogai5l8.xlsx"')
        os.system(f'wget -q -O {excel_file_table_benchmark_eth}'
                  f' -N "https://ethercalc.org/bzn1f11s4w2b.xlsx"')
    temp_output_file = os.path.join(opentraj_root, "README__temp.md")

    table_html_format = False
    if '--build' in sys.argv:
        output_str_ = ''
        print(opentraj_root)

        build_readme()
        print(output_str_)

        with open(temp_output_file, "w") as out_file:
            out_file.write(output_str_)

    if '--confirm' in sys.argv and os.path.exists(temp_output_file):
        with open(temp_output_file) as f_in:
            lines = f_in.readlines()
            with open(os.path.join(opentraj_root, "README.md"), "w") as f_out:
                f_out.writelines(lines)
        os.remove(temp_output_file)

