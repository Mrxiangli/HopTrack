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

for idx, each in enumerate(mot_files):
    df = pd.read_csv(each)
    worksheet.write(idx+1, 0, df['seq'][0])
    worksheet.write(idx+1, 1, df['CLR_FP'][0])
    worksheet.write(idx+1, 2, df['CLR_FN'][0])
    worksheet.write(idx+1, 3, df['IDSW'][0])
    worksheet.write(idx+1, 4, df['GT_Dets'][0])
    worksheet.write(idx+1, 5, df['IDF1'][0])
    worksheet.write(idx+1, 6, df['MOTA'][0])
    total_FP += df['CLR_FP'][0]
    total_FN += df['CLR_FN'][0]
    total_ID += df['IDSW'][0]
    total_GT += df['GT_Dets'][0]

mota = 1-(total_FN+total_FP+total_ID)/total_GT
worksheet.write(idx+1, 0, "summary")
worksheet.write(idx+1, 1, total_FP)
worksheet.write(idx+1, 2, total_FN)
worksheet.write(idx+1, 3, total_ID)
worksheet.write(idx+1, 4, total_GT)

worksheet.write(idx+1, 6, mota)

workbook.close()