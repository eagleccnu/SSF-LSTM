import argparse
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import utils


def load_model(json_path, weights_path):

    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)

    return loaded_model


def load_data(x_path, y_path):

    X = np.load(x_path, allow_pickle=True).item()
    Y = np.load(y_path, allow_pickle=True).item()

    return X, Y


def plot_scatter(real_b_all, pre_b_all, pcc_avg):

    plt.figure(figsize=(6, 5))
    plt.xlim(-3, 13)
    plt.ylim(-3, 13)
    plt.scatter(real_b_all, pre_b_all, s=0.5)
    plt.plot([min(real_b_all), max(real_b_all)], [min(real_b_all), max(real_b_all)], '-', c='black')
    plt.xlabel('Experiment of B-factor', fontsize=15)
    plt.ylabel('Prediction of B-factor', fontsize=15)
    plt.title(f'Average PCC = {pcc_avg:.2f}', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def plot_comparison(real_b, pre_b, sequence_name, pcc):

    plt.figure(figsize=(14, 8))
    plt.title('PDB ID: ' + sequence_name + ',  Length: ' + str(len(real_b)) + ',  PCC: ' + f'{pcc:.2f}', fontsize=24)
    plt.plot(range(1, len(real_b) + 1), real_b, '-', c='b', label='Experiment', linewidth=2)
    plt.plot(range(1, len(pre_b) + 1), pre_b, '-', c='r', label='Predict', linewidth=2)
    plt.xlabel('Residue', fontsize=26)
    plt.ylabel('B-factor', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict B-factor values and optionally plot comparisons.')
    parser.add_argument('--pdbID', type=str, help='PDB ID of the protein (e.g., 7DTL)')
    parser.add_argument('--chainID', type=str, help='Chain ID of the protein (e.g., A)')
    args = parser.parse_args()

    json_path = 'modelP.json'
    weights_path = "modelW.h5"
    x_path = 'X_650.npy'
    y_path = 'Y_650.npy'

    model = load_model(json_path, weights_path)
    X_650, Y_650 = load_data(x_path, y_path)

    sequence_names = list(X_650.keys())
    pcc_ls = []
    real_b_all = []
    pre_b_all = []

    ls = {}
    if args.pdbID and args.chainID:

        # Predict and plot for a single sequence
        get_features = utils.Protein_bfactor(args.pdbID, args.chainID)
        get_features.get_data_from_pdb()

        aa, pc, phi_psi_area, reso_R = get_features.aa_pc_phi_psi_area_reso_R()
        hmm, pssm = get_features.get_pssm_hmm()
        pd = get_features.get_packingdensity()

        res_features = np.concatenate((aa,  pc, hmm, pssm, phi_psi_area, pd, reso_R), axis=1)
        x_test, y_test = get_features.preprocess_data(res_features)

        y_test = [(i - np.mean(y_test)) / np.std(y_test) for i in y_test]

        y_pre = model.predict(x=x_test, batch_size=10, verbose=1)
        y_pre = [item[0] for item in y_pre]

        pcc = np.corrcoef(y_test, y_pre)[0][1]

        # print(f'Normalized experimental B-factor values: {y_test}')
        # print(f'Predicted B-factor values: {y_pre}')
        print(f'PCC: {round(float(pcc), 2)}')

        plot_comparison(y_test, y_pre, args.pdbID + args.chainID, pcc)

    else:
        # Predict all sequences
        for sequence_name in sequence_names:
            x_test = X_650[sequence_name]
            y_test = Y_650[sequence_name]
            y_test = [(i - np.mean(y_test)) / np.std(y_test) for i in y_test]

            y_pre = model.predict(x=x_test, batch_size=10, verbose=1)
            y_pre = [item[0] for item in y_pre]

            pcc = np.corrcoef(y_test, y_pre)[0][1]
            pcc_ls.append(pcc)

            real_b_all.extend(y_test)
            pre_b_all.extend(y_pre)

            ls[sequence_name] = pcc

        pcc_mean = np.mean(pcc_ls)

        print(f'Average PCC on test set: {round(float(pcc_mean), 2)}')

        plot_scatter(real_b_all, pre_b_all, pcc_mean)

        top_4_items = sorted(ls.items(), key=lambda  item: item[1], reverse=True)[:4]
        for key, value in top_4_items:
            print(key, value)


if __name__ == '__main__':
    main()




