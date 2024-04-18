import pickle

from PROD import featureClassifier

if __name__ == "__main__":

    with open("featureClassifier17042024.pkl", 'rb') as f:
        loaded_object = pickle.load(f)
        c = featureClassifier()

    #c.save_params("classifier6dim_new.pkl")

    print()
