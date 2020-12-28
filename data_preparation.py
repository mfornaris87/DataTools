import pandas as pd
import numpy as np
import scipy.sparse
import scipy
import glob


def disorder_labels():
    """
        Read the list of disorder labels from a file
    :return: Python list
    """
    disorder_list = []
    with open('./data/disorder_labels.txt', mode='r') as f:
        for line in f:
            disorder_list.append(line.strip('\n'))
    return disorder_list


def get_training_proteins():
    """
        From the dataframe of disprot + control proteins it returns the training proteins
    :return: Python list
    """
    cols = ['Entry']
    for item in disorder_labels():
        cols.append(item)
    df = pd.read_csv('./data/cleaned_training_do.tsv', sep='\t', usecols=cols)
    # df.to_csv('./data/cleaned_training_do.tsv', sep='\t', index=False)
    return df['Entry'].to_list()


def get_all_proteins():
    """
            From the dataframe of disprot + control + target proteins it returns all proteins
        :return: Python list
        """
    cols = ['Entry']
    for item in disorder_labels():
        cols.append(item)
    df = pd.read_csv('./data/cleaned_predicting_do.tsv', sep='\t', usecols=cols)
    # df.to_csv('./data/cleaned_training_do.tsv', sep='\t', index=False)
    return df['Entry'].to_list()


def get_target_proteins():
    """
        Read the list of human targets to predict from a file
    :return: Python list
    """
    targets_list = []
    with open('./data/human_targets.txt', mode='r') as f:
        for line in f:
            targets_list.append(line.strip('\n'))
    return targets_list


def disprot_proteins():
    """
        Read the list of all disprot proteins from a file
    :return: Python list
    """
    df = pd.read_csv('./data/final_occurrence_df_with_moveup_cleaned.tsv', sep='\t')
    proteins = df['prot_id'].to_list()
    return proteins


def to_coordinates_one(matrix, matrix_index, matrix_columns, path):
    """
        This function converts a Dataframe of n rows by m columns into a Dataframe of n X m rows and 3 columns,
        the columns are the row index, the column index and the value in the cell.
    :param matrix: Pandas Dataframe
    :param matrix_index: Output of the dataframe['Entry] or dataframe['target']
    :param matrix_columns: Output of dataframe.columns
    :param path: Place to save the resulting Dataframe
    :return: Pandas Dataframe
    """
    x_list = matrix_index.to_list()
    # print(x_list)
    y_list = matrix_columns.to_list()
    indexer = y_list.pop(0)
    # print(indexer)
    # print(len(y_list))
    # print(y_list)
    coordinates_matrix = pd.DataFrame()
    count = 0
    for x in x_list:
        row = matrix.loc[matrix[indexer] == x]
        for y in y_list:
            temp = pd.DataFrame({'x': [x_list.index(x)], 'y': [y_list.index(y)], 'value': row.loc[row[indexer] == x][y]})
            coordinates_matrix = pd.concat([coordinates_matrix, temp], ignore_index=True)
    coordinates_matrix.to_csv(path, sep='\t', index=False)
    return coordinates_matrix


def to_coordinates(matrix, matrix_index, matrix_columns):
    """
        This function converts a Dataframe of n rows by m columns into a Dataframe of n X m rows and 3 columns,
        the columns are the row index, the column index and the value in the cell. Este algoritmo funciona dividiendo
        en varios archivos el dataframe resultante
    :param matrix: Pandas Dataframe
    :param matrix_index: Output of the dataframe['Entry] or dataframe['target']
    :param matrix_columns: Output of dataframe.columns
    :return: Pandas Dataframe
    """
    x_list = matrix_index.to_list()
    # print(x_list)
    y_list = matrix_columns.to_list()
    indexer = y_list.pop(0)
    # print(indexer)
    # print(len(y_list))
    # print(y_list)
    coordinates_matrix = pd.DataFrame()
    count = 0
    for x in x_list:
        row = matrix.loc[matrix[indexer] == x]
        for y in y_list:
            count += 1
            if count % 2475 == 0:
                path = './coo_matrix/file' + str(count / 2475) + '.tsv'
                coordinates_matrix.to_csv(path, sep='\t', index=False)
                coordinates_matrix = pd.DataFrame()
                # print(coordinates_matrix)
                print('Your reached ', count, ' lines')
            else:
                temp = pd.DataFrame({'x': [x_list.index(x)], 'y': [y_list.index(y)], 'value': row.loc[row[indexer] == x][y]})
                coordinates_matrix = pd.concat([coordinates_matrix, temp], ignore_index=True)
    if count % 2475 != 0:
        path = './coo_matrix/file' + str(count / 2475) + '.tsv'
        coordinates_matrix.to_csv(path, sep='\t', index=False)


def to_sparse(my_dataframe):
    """
        Turns every dataframe in coordinates format to a scipy sparse matrix
    :param my_dataframe: Pandas dataframe with only 3 columns: x, y and value
    :return: Scipy sparse matrix
    """
    # df = pd.read_csv(file_name, sep='\t')
    # macau_data = pd.read_csv('./data/coo_main_matrix_original.tsv', sep='\t')
    # macau_data = main_matrix_to_coordinates()
    my_dataframe = my_dataframe.astype({'x': int, 'y': int, 'value': float})
    sparse_matrix = scipy.sparse.csr_matrix((my_dataframe.value, (my_dataframe.x, my_dataframe.y)))
    return sparse_matrix


def cleaning_ytest(ytest):
    """
        This function cleans the first 10% of the dataframe observations
    :param ytest: Pandas dataframe to clean
    :return: Cleaned pandas dataframe
    """
    ytest.to_csv('./data/temp_dataframe.tsv', sep='\t', index=False)
    row10percent = int((len(ytest) * 10) / 100)
    read_Ytest = pd.read_csv('./data/temp_dataframe.tsv', usecols=['x', 'y', 'value'], sep='\t',
                             nrows=(row10percent - 1))
    columns = read_Ytest.columns
    empty_ytest = pd.DataFrame(columns=columns)
    empty_ytest['x'] = read_Ytest['x']
    empty_ytest['y'] = read_Ytest['y']

    df = pd.read_csv('./data/temp_dataframe.tsv', sep='\t', header=None, skiprows=row10percent)
    df.columns = ['x', 'y', 'value']
    df = df.astype({'x': int, 'y': int, 'value': float})
    empty_ytest = pd.concat([empty_ytest, df])
    empty_ytest.index = range(empty_ytest.shape[0])
    return empty_ytest


def select_good_testing_proteins():
    """
        This function checks if the bitscore of each target protein against those of Disprot if it is over 500,
        then it is likely they are the same protein, so we create a new dataframe with those proteins to use as
        validating set, given their disorder is already known.
    :return: Pandas DataFrame
    """
    good_proteins = pd.DataFrame()
    path = './bitscores1/'
    all_files = glob.glob(path + '*.tsv')
    # read all files contents as pandas dataframes into a list
    df_list = [pd.read_table(file, sep='\t') for file in all_files]
    # concatenate them together
    big_df = pd.concat(df_list)

    for prot in disprot_proteins():
        row = big_df.loc[big_df[str(prot)] >= 500]
        good_proteins = pd.concat([good_proteins, row], ignore_index=True)
    # good_proteins.to_csv('./data/good_proteins.tsv', sep='\t', index=False)
    return good_proteins


def get_top_classes():
    top_classes_l = ['DO:00001', 'DO:00008', 'DO:00017', 'DO:00024', 'DO:00035', 'DO:00040', 'DO:00050', 'DO:00056',
                     'DO:00063', 'DO:00064', 'DO:00071', 'DO:00072', 'DO:00074', 'DO:00076']
    df = pd.read_csv('./data/training_do.tsv', sep='\t', index_col=0)
    # print(df)
    cols = df.columns
    # print(cols)
    labels = cols.to_list().copy()
    top_classes = {}
    labels.pop(0)
    for label in top_classes_l:
        top_classes[labels.index(label)] = label
    return top_classes


def get_middle_classes():
    middle_classes_l = ['DO:00002', 'DO:00009', 'DO:00010', 'DO:00018', 'DO:00021', 'DO:00025', 'DO:00065']
    df = pd.read_csv('./data/training_do.tsv', sep='\t', index_col=0)
    cols = df.columns
    dis_labels = cols.to_list().copy()
    middle_classes = {}
    for label in middle_classes_l:
        middle_classes[dis_labels.index(label)] = label
    return middle_classes


def translate_protein_disorder_test(dataset, path_output):
    """
    Translate a test dataframe into another with one column for the protein index and another for the disorder index
    :param dataset: Pandas dataframe with 3 columns: x, y and value
    :param path_output: Path of the file to store the result dataframe
    :return: Pandas dataframe with 2 columns: entry, disorder
    """
    disorder_protein = pd.DataFrame()
    for idx, row in dataset.iterrows():
        if row[2] == 1:
            temp = pd.DataFrame({'entry': [int(row[0])], 'disorder': [int(row[1])]})
        else:
            temp = pd.DataFrame({'entry': [int(row[0])], 'disorder': [int(21)]})
        disorder_protein = pd.concat([disorder_protein, temp], ignore_index=True)
    disorder_protein.to_csv(path_output, sep='\t', index=False)
    return disorder_protein


def translate_protein_disorder_pred(dataset, path_output):
    """
        Translate a Macau predictions dataframe into another with 4 columns: entry, real_disorder, predicted_disorder
        and y_pred_std using the indices of the proteins and disorder labels according to the original dataframe
        :param dataset: Pandas dataframe with 3 columns: x, y and value
        :param path_output: Path of the file to store the result dataframe
        :return: Pandas dataframe with 4 columns: entry, real_disorder, predicted_disorder
        and y_pred_std using the indices of the proteins and disorder labels according to the original dataframe
        """
    disorder_protein = pd.DataFrame()
    for idx, row in dataset.iterrows():
        if np.round(row[2]) == 1 and np.round(row[3]) == 1:
            temp = pd.DataFrame(
                {'entry': [int(row[0])], 'real_disorder': [int(row[1])],
                 'predicted_disorder': [int(row[1])], 'y_pred_std': [row[4]]})
        elif np.round(row[2]) == 1 and np.round(row[3]) == 0:
            temp = pd.DataFrame(
                {'entry': [int(row[0])], 'real_disorder': [int(row[1])],
                 'predicted_disorder': [int(21)], 'y_pred_std': [row[4]]})
        elif np.round(row[2]) == 0 and np.round(row[3]) == 1:
            temp = pd.DataFrame({'entry': [int(row[0])], 'real_disorder': [int(21)],
                                 'predicted_disorder': [int(row[1])], 'y_pred_std': [row[4]]})
        else:
            temp = pd.DataFrame({'entry': [get_training_proteins()[int(row[0])]], 'real_disorder': [int(21)],
                                 'predicted_disorder': [int(21)], 'y_pred_std': [row[4]]})
        disorder_protein = pd.concat([disorder_protein, temp], ignore_index=True)
    disorder_protein.to_csv(path_output, sep='\t', index=False)
    return disorder_protein


def final_disorder_predictions():
    """
    In the targest prediction it is necessary the disprot label for no disorder, it is DO:00079
    and it is appended to the previous predicted disorder on this function
    :return: Python list
    """
    return disorder_labels().append('DO:00079')



def clean_training_matrix():
    """
        Reshape and return the training matrix from a file
    :return: Pandas dataframe
    """
    df = pd.read_csv('./data/predicting_do.tsv', sep='\t')
    cols = df.columns.to_list()
    cols.pop(0)
    new_df = pd.read_csv('./data/predicting_do.tsv', sep='\t', usecols=cols)
    new_df.to_csv('./data/cleaned_predicting_do.tsv', sep='\t', index=False)
    return new_df


def create_training_matrix():
    """
        Create a training matrix with only the disprot proteins
    :return:
    """
    humans = pd.read_csv('./data/final_occurrence_df_with_moveup.tsv', sep='\t',
                         usecols=['prot_id', 'DO:00001', 'DO:00002', 'DO:00008', 'DO:00009',
                                  'DO:00010', 'DO:00017', 'DO:00018', 'DO:00021', 'DO:00024', 'DO:00025',
                                  'DO:00035', 'DO:00040', 'DO:00050', 'DO:00056', 'DO:00063', 'DO:00064',
                                  'DO:00065', 'DO:00071', 'DO:00072', 'DO:00074', 'DO:00076'])
    humans.to_csv('./data/all_disprot_training.tsv', sep='\t', index=False)
    return humans


def create_target_matrix():
    """
        Create the human targets matrix
    :return: Pandas dataframe
    """
    cols = ['prot_id']
    for item in disorder_labels():
        cols.append(item)
    proteins = pd.Series(get_target_proteins())
    targets = pd.DataFrame(columns=cols)
    targets['prot_id'] = proteins
    targets.to_csv('./data/targets_matrix.tsv', sep='\t', index=False)
    return targets


def reshape_new_bitscores():
    cols = ['target']
    for prot in get_training_proteins():
        cols.append(prot)
    df = pd.read_csv('./data/merged_training_bitscore_df.tsv', sep='\t', index_col=-1, usecols=cols)
    df.to_csv('./data/reordered_merged_training_bitscore_df.tsv', sep='\t')


def main_matrix_to_coordinates():
    """
        This method turns the matrix of disprot training proteins into a coordinates matrix with only 3 columns:
        x, y and value
    :return: Pandas dataframe
    """
    my_humans = create_training_matrix()
    # Convert the original matrix to coordinates
    # The y will be the columns and they are the disorder labels
    y_list = my_humans.columns.tolist()
    y_list = y_list[1:]
    main_matrix = pd.DataFrame()
    # Each protein is a row in the dataframe
    proteins = my_humans['prot_id'].to_list()
    for prot in proteins:
        row = my_humans.loc[my_humans['prot_id'] == str(prot)]
        for label in y_list:
            temp = pd.DataFrame({'x': [proteins.index(prot)], 'y': [y_list.index(label)],
                                 'value': row.loc[row['prot_id'] == str(prot)][str(label)]})
            main_matrix = pd.concat([main_matrix, temp])
    main_matrix.index = range(main_matrix.shape[0])
    main_matrix.astype({'x': int, 'y': int, 'value': float})
    main_matrix.to_csv('./data/coo_main_matrix_original.tsv', sep='\t', index=False)
    return main_matrix


def get_disprot_bitscores():
    """
        Create a new bitscore matrix with only the proteins in disprot
    :return: Pandas dataframe
    """
    bitscores_cols = ['target']
    for prot in disprot_proteins():
        bitscores_cols.append(prot)
    all_bitscores = pd.read_csv('./data/merged_all_dp_and_human_targets_bitscore_df.zip', sep='\t', index_col=-1,
                                usecols=bitscores_cols)
    all_bitscores.to_csv('./data/reordered_merged_all_dp_and_human_targets_bitscore_df.tsv', sep='\t')
    reordered_bitscores = pd.read_csv('./data/reordered_merged_all_dp_and_human_targets_bitscore_df.tsv', sep='\t')
    # print(reordered_bitscores)
    bitscores = pd.DataFrame()
    for prot in disprot_proteins():
        temp = pd.DataFrame(reordered_bitscores.loc[reordered_bitscores['target'] == str(prot)])
        bitscores = pd.concat([bitscores, temp])
    bitscores.index = range(bitscores.shape[0])
    bitscores.to_csv('./data/all_disprot_bitscores.tsv', sep='\t', index=False)
    return bitscores


def disprot_bitscores_coo():
    """
        Takes the bitscore matrix of the disprot proteins and turns it into a coordinates matrix with only 3 columns:
        x, y and value. Where x is the row, y is the column and the value of the cell
    :return:
    """
    disprot_bitscore = pd.read_csv('./data/all_disprot_bitscores.tsv', sep='\t', index_col=0)
    # print(disprot_bitscore)
    bitscore_coordinates = pd.DataFrame()
    for i in range(len(disprot_proteins())):
        row = disprot_bitscore.iloc[i]
        for j in range(len(disprot_proteins())):
            temp = pd.DataFrame({'x': [i], 'y': [j], 'value': row.iloc[j]})
            bitscore_coordinates = pd.concat([bitscore_coordinates, temp])
    bitscore_coordinates.index = range(bitscore_coordinates.shape[0])
    bitscore_coordinates.to_csv('./data/disprot_bitscores_coordinates.tsv', sep='\t', index=False)
    print(bitscore_coordinates)
