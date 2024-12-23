from dynamic_similarity_algorithm import Dynamic_Similarity_Algorithm

main_dir = '../content/drive/MyDrive/hpml_final_project/D1/'
x=0
print('Enter desired recall:')
x = input()

sg = Dynamic_Similarity_Algorithm(budget=5631064, delimiter='\t',  sourceFilePath= main_dir + "SourceDataset.csv", targetFilePath = main_dir + "TargetDataset.csv", testFilePath = main_dir + 'similarity_results.csv', target_recall = float(x), similarity_index_range = 0.1)
sg.applyProcessing()
