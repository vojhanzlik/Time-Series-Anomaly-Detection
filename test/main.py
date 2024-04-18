import pickle


if __name__ == "__main__":

    with open("deviationClassifier.pkl", 'rb') as f:
        loaded_object = pickle.load(f)
        print()

    #c.save_params("classifier6dim_new.pkl")


