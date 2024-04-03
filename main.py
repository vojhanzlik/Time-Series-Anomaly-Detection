import pickle

from classifier_model.classifier import FeatureClassifier1

if __name__ == "__main__":

    with open("classifier6dim_new.pkl", 'rb') as f:
        loaded_object = pickle.load(f)
        c = FeatureClassifier1()
        #c.naive_fit(loaded_object)

    #c.save_params("classifier6dim_new.pkl")

    print()
