import os
from config import Config

import json

import dill
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

BOT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
print("[DEBUG] BOT_DIR_PATH >> {}".format(BOT_DIR_PATH))


class BotSkeleton(object):
    def __init__(self, bot_id, bot_name, params):

        self.id = bot_id
        self.name = bot_name
        self.cT, self.dT, self.n_intents = params
        self.ready = False

        # create folder for saving model later
        self._make_home_folder()

    def update_params(self, params):
        self.cT, self.dT, self.n_intents = params

    def predict(self, message):

        if self.ready:
            p_probas = np.array(self.model.predict_proba([message])[0])
            print("[INFO] predicted probability : {}".format(max(p_probas)))

            if max(p_probas) >= self.cT:
                # return self.lb2id[self.model.classes_[np.argmax(p_probas)]]
                json_obj = {
                    "return_type": True,
                    "data": [self.lb2id[self.model.classes_[np.argmax(p_probas)]]]
                }
                return json_obj
            else:
                # in case have only one label
                if len(p_probas) < 2:
                    # return [self.lb2id[p_probas[0]]]
                    json_obj = {
                        "return_type": False,
                        "data": [self.lb2id[p_probas[0]]]
                    }
                    return json_obj

                sorted_indice = p_probas.argsort()[::-1]

                if p_probas[sorted_indice[0]] >= p_probas[sorted_indice[1]] + self.dT:
                    # return self.lb2id[self.model.classes_[sorted_indice[0]]]
                    json_obj = {
                        "return_type": True,
                        "data": [self.lb2id[self.model.classes_[sorted_indice[0]]]]
                    }
                    return json_obj
                else:
                    n = min(len(p_probas), self.n_intents)
                    similarities = [self.lb2id[sorted_indice[i]] for i in range(n)]
                    json_obj = {
                        "return_type": False,
                        "data": similarities
                    }
                    return json_obj
        else:
            # return -1
            json_obj = {
                "return_type": False,
                "data": []
            }
            return json_obj

    def fit(self, X, ids, model_version):
        try:
            # training model
            y, label2id = self._ids2labels(ids)

            pipeline = Pipeline([
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("clf", RandomForestClassifier())
            ])

            model = pipeline.fit(X, y)

            # save model file and label2id dict
            model_path = os.path.join(self.home_folder, model_version)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            path_to_model = os.path.join(model_path, Config.MODEL_PKL)
            with open(path_to_model, 'wb') as model_file:
                dill.dump(model, model_file)

            with open(os.path.join(model_path, Config.LB2ID_PKL), 'wb') as lb2id_file:
                dill.dump(label2id, lb2id_file)

            file_size = os.path.getsize(path_to_model)

            return (True, path_to_model, file_size)

        except:
            print("[ERROR] traing data error with bot {}", self.id)
            return (False, "", 0)

    def reload_model(self, model_version):
        try:
            # load model and label to id dict
            with open(os.path.join(self.home_folder, model_version, Config.MODEL_PKL), 'rb') as model_file:
                self.model = dill.load(model_file)

            with open(os.path.join(self.home_folder, model_version, Config.LB2ID_PKL), 'rb') as lb2id_file:
                self.lb2id = dill.load(lb2id_file)

            # active bot
            self.ready = True
            return True

        except:
            self.ready = False
            return False

    def turn_off(self):
        self.ready = False

    def _make_home_folder(self):
        bot_home_folder = os.path.join(BOT_DIR_PATH, Config.MODELS_FOLDER, self.id)
        if not os.path.exists(bot_home_folder):
            os.makedirs(bot_home_folder)
        self.home_folder = bot_home_folder

    def _ids2labels(self, ids):
        y = []
        label2id = {}
        id2label = {}

        id_set = set(ids)

        print("[DEBUG] id set : {}".format(id_set))

        for idx, id in enumerate(id_set):
            label2id[idx] = id
            id2label[id] = idx

        for id in ids:
            y.append(id2label[id])

        return (y, label2id)


class BotsManager(object):
    def __init__(self, bot_ver_dict={}):
        """

        :param bot_ver_dict: bot_id and current working version getting from backend
        """
        self.bot_dict = {}

        
        # p = os.path.join(BOT_DIR_PATH, Config.MODELS_FOLDER)
        # bot_ids = [f.name for f in os.scandir(p) if f.is_dir()]
        # for bot_id in bot_ids:
        #     self.add_new_bot(bot_id)
        #     bot_home_folder = os.path.join(BOT_DIR_PATH, Config.MODELS_FOLDER, bot_id)
        #     versions = [f.name for f in os.scandir(bot_home_folder) if f.is_dir()]
        #     if len(versions) > 0:
        #         self.bot_dict[bot_id].reload_model(versions[-1])
        #         print("[INFO] re-activate bot {0}, version {1}".format(bot_id, versions[-1]))

        if len(bot_ver_dict) > 0:
            p = os.path.join(BOT_DIR_PATH, Config.MODELS_FOLDER)
            bot_ids = [f.name for f in os.scandir(p) if f.is_dir()]
            for bot_id in bot_ids:
                if bot_id in bot_ver_dict:
                    self.add_new_bot(bot_id)
                    bot_home_folder = os.path.join(BOT_DIR_PATH, Config.MODELS_FOLDER, bot_id)
                    versions = [f.name for f in os.scandir(bot_home_folder) if f.is_dir()]
                    if (bot_ver_dict[bot_id] is not None) and (bot_ver_dict[bot_id] in versions):
                        self.bot_dict[bot_id].reload_model(bot_ver_dict[bot_id])
                        print("[INFO] re-activate bot {0}, version {1}".format(bot_id, bot_ver_dict[bot_id]))


    def add_new_bot(self, bot_id, bot_name="", params=(0.65, 0.02, 3)):
        b = BotSkeleton(bot_id, bot_name, params)
        self.bot_dict[bot_id] = b

    def update_params(self, bot_id, params):
        self.bot_dict[bot_id].update_params(params)

    def predict(self, bot_id, message):
        if bot_id not in self.bot_dict:
            print("[ERROR] not found bot with id {}".format(bot_id))
            return -2

        return self.bot_dict[bot_id].predict(message)

    def fit(self, bot_id, X, y, model_version):
        if bot_id not in self.bot_dict:
            print("[ERROR] not found bot with id {}".format(bot_id))
            return (False, "", 0)

        return self.bot_dict[bot_id].fit(X, y, model_version)

    def reload_model(self, bot_id, model_version):
        if bot_id not in self.bot_dict:
            print("[ERROR] not found bot with id {}".format(bot_id))
            return False

        return self.bot_dict[bot_id].reload_model(model_version)

    def turn_off(self, bot_id):
        if bot_id not in self.bot_dict:
            print("[ERROR] not found bot with id {}".format(bot_id))
            return False

        self.bot_dict[bot_id].turn_off()
        return True
