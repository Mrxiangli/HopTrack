import pandas as pd
import xlsxwriter


mot_files = [
    'MOT16-02-evaluation.csv',
    'MOT16-04-evaluation.csv',
    'MOT16-05-evaluation.csv',
    'MOT16-09-evaluation.csv',
    'MOT16-10-evaluation.csv',
    'MOT16-11-evaluation.csv',
    'MOT16-13-evaluation.csv',
]

# mot_files = [
#     'MOT20-01-evaluation.csv',
#     'MOT20-02-evaluation.csv',
#     'MOT20-03-evaluation.csv',
#     'MOT20-05-evaluation.csv',
# ]

# mot_files = [
#     'KITTI-13-evaluation.csv',
#     'KITTI-17-evaluation.csv',
# ]


workbook = xlsxwriter.Workbook("MOT16_mota.xlsx")
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "sequence")
worksheet.write(0, 1, "FP")
worksheet.write(0, 2, "FN")
worksheet.write(0, 3, "IDSW")
worksheet.write(0, 4, "GT_Dets")
worksheet.write(0, 5, "IDF1")
worksheet.write(0, 6, "MOTA")

total_FP = 0
total_FN = 0
total_ID = 0
total_GT = 0

ct=1
for idx, each in enumerate(mot_files):
    df = pd.read_csv(each)
    worksheet.write(ct, 0, df['seq'][0])
    worksheet.write(ct, 1, df['CLR_FP'][0])
    worksheet.write(ct, 2, df['CLR_FN'][0])
    worksheet.write(ct, 3, df['IDSW'][0])
    worksheet.write(ct, 4, df['GT_Dets'][0])
    worksheet.write(ct, 5, df['IDF1'][0])
    worksheet.write(ct, 6, df['MOTA'][0])
    total_FP += df['CLR_FP'][0]
    total_FN += df['CLR_FN'][0]
    total_ID += df['IDSW'][0]
    total_GT += df['GT_Dets'][0]
    ct+=1

mota = 1-(total_FN+total_FP+total_ID)/total_GT
worksheet.write(ct, 0, "summary")
worksheet.write(ct, 1, total_FP)
worksheet.write(ct, 2, total_FN)
worksheet.write(ct, 3, total_ID)
worksheet.write(ct, 4, total_GT)

worksheet.write(ct, 6, mota)

workbook.close()