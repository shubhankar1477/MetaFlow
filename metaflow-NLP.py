from metaflow import FlowSpec,step,Parameter,IncludeFile
import utils


def script_path(filename):
    import os
    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath,filename)


class NLPFlow(FlowSpec):
    fname = IncludeFile(
        "fname",
        help="The path to a movie metadata file.",
        default=script_path("data/new_data.csv"),
    )


    @step
    def start(self):
        import pandas as pd
        from io import StringIO
        self.data = pd.read_csv(StringIO(self.fname))
        self.next(self.prepare_data)
    @step
    def prepare_data(self):
        self.data['label'] = self.data['sentiment'].map({"positive":1,"negative":0})
        self.data['review'] = self.data['review'].apply(utils.transform)
        print("Data Prepared")
        self.next(self.tokenise_split)

    @step 
    def tokenise_split(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split

        try:
            print("Split and TOkenising")
            self.tfidf = TfidfVectorizer(use_idf=True)
            self.docs = list(self.data['review'])
            self.docs = self.tfidf.fit_transform(self.docs)
            self.X = self.docs.toarray()
            self.y=self.data['label']
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X, self.y, test_size=0.2, random_state=123, stratify=self.y)
            
        except Exception as e:
            print(e)
        self.next(self.fit_model_1,self.fit_model_2)
    @step
    def fit_model_1(self):
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(random_state=1)
        rf_model.fit(self.X_train,self.y_train)
        self.prediction_1 = rf_model.predict(self.X_test)
        self.next(self.join)

    @step
    def fit_model_2(self):
        from sklearn.naive_bayes import GaussianNB
        nb_model = GaussianNB()
        nb_model.fit(self.X_train,self.y_train)
        self.prediction_2 = nb_model.predict(self.X_test)
        self.next(self.join)

    @step
    def join(self,inputs):
        self.merge_artifacts(inputs)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        from sklearn.metrics import accuracy_score
        print("Accuracy for RF {}".format(accuracy_score(self.prediction_1,self.y_test)))
        print("Accuracy for Naive {}".format(accuracy_score(self.prediction_2,self.y_test)))
        self.next(self.end)

    @step
    def end(self):
        print("finished")

if __name__ == '__main__':
    NLPFlow()