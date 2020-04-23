from pandas import DataFrame, read_csv
import pandas as pd
import xlrd
import os


# FIXME
opentraj_root = "/home/cyrus/workspace2/OpenTraj"
input_readme = os.path.join(opentraj_root, "README.md")
output_readme = os.path.join(opentraj_root, "README_new.md")

output_str = ''
# Todo: find beginning and end of table 1
table1_begin = '<!----Table1-begin---->'
table1_end = '<!----Table1-end---->'

# Todo: Add texts before table 1
text_before_table1 = ''
output_str += text_before_table1

# FIXME: Build Table 1
table1_file = os.path.join(opentraj_root, "tools/doc/opentraj-table1.xls")
df = pd.read_excel(table1_file, header=[0])

content = df.values.tolist()
headers = df.axes[1].values.tolist()
header_inds = [0, 1, 6, 7, 8]  # ['Sample', 'Name', 'Description', 'REF']

num_rows = len(content)
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
        # aa = str(content[ii][jj])
        item_ij = content[ii][jj] if str(content[ii][jj]) != 'nan' else ' '
        content_str += str(item_ij).replace('|', '') + ' | '
    content_str += '\n'


print(header_str + '\n' + seperator_str + '\n' + content_str)
output_str += table1_begin + '\n' + header_str + '\n' + seperator_str + '\n' + content_str + '\n' + table1_end + '\n'


# Todo: find beginning and end of table 2
# Todo: Add texts before table 2

# Todo: Build Table ETH
table2_file = os.path.join(opentraj_root, "tools/doc/ETH-table.xls")


# Todo: Append rest of the text
with open(output_readme, 'w') as out_file:
    out_file.write(output_str)