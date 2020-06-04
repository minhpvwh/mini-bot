from config import Config
from bot.base import BotsManager
from utils import norm_string

import json

from flask import Flask, request
from flask_restful import Api, Resource

app = Flask(__name__)
app.config.from_object(Config)
api = Api(app)

bots_manager = BotsManager()


class CreateBotHandler(Resource):
    def get(self):
        return "[INFO] CreateBotHandler is still alive ..."

    def post(self):
        print("[DEBUG] Get request creating new bot ...")
        json_data = request.get_json()
        bot_id, bot_name, params = self._parse_json(json_data)

        bots_manager.add_new_bot(bot_id, bot_name, params)

        return True

    def _parse_json(self, json_data):
        bot_name = json_data["topicName"]
        bot_id = json_data["id"]
        cT = float(json_data["generalIntentThresholdConfidence"])
        dT = float(json_data["generalQaIntentThresholdDiff"])
        n_intents = int(json_data["generalNumberIntent"])

        return (bot_id, bot_name, (cT, dT, n_intents))


class ChatHandler(Resource):
    def get(self):
        return "[INFO] ChatHandler is still alive ..."

    def post(self):
        json_data = request.get_json()

        bot_id = json_data["id"]
        message = json_data["msg"]

        return bots_manager.predict(bot_id, message)


class RetrainHandler(Resource):
    def get(self):
        return "[INFO] RetrainHandler is still alive ..."

    def post(self):
        print("[DEBUG] Get request training bot ...")
        json_data = request.get_json()

        print("---------------------------")
        print(json_data)
        print("---------------------------")

        # TODO parsing data
        bot_id, texts, ids, model_version = self._parse_json(json_data)
        (status, path_to_model,  file_size) = bots_manager.fit(bot_id, texts, ids, model_version)
        json_txt = {
            "status": status,
            "path_to_model": path_to_model,
            "file_size": str(file_size) + " bytes"
        }
        return json_txt
        # return json.dumps(json_txt)

    def _parse_json(self, json_data):
        bot_id = json_data["id"]
        model_version = json_data["version"]

        texts = []
        ids = []
        for item in json_data["data"]:
            intent_id = item["id"]
            for e in item["intentEntrace"]:
                texts.append(norm_string(e["value"]))
                ids.append(intent_id)

        return (bot_id, texts, ids, model_version)
        # bots_manager.fit(bot_id, texts, ids, model_version)


class UpdateParamsHandler(Resource):
    def get(self):
        return "[INFO] UpdateParamsHandler is still alive ..."

    def post(self):
        print("[DEBUG] Get request updating params ...")
        json_data = request.get_json()
        bot_id, params = self._parse_json(json_data)
        bots_manager.update_params(bot_id, params)

    def _parse_json(self, json_data):
        bot_id = json_data["id"]
        cT = float(json_data["generalIntentThresholdConfidence"])
        dT = float(json_data["generalQaIntentThresholdDiff"])
        n_intents = int(json_data["generalNumberIntent"])

        return (bot_id, (cT, dT, n_intents))


class TurnOnHandler(Resource):
    def get(self):
        return "[INFO] OnOffHandler is still alive ..."

    def post(self):
        json_data = request.get_json()

        bot_id = json_data["bot_id"]
        model_version = json_data["model_version"]

        return bots_manager.reload_model(bot_id, model_version)

class TurnOffHandler(Resource):
    def get(self):
        pass

    def post(self):
        json_data = request.get_json()
        bot_id = json_data["bot_id"]

        bots_manager.turn_off(bot_id)

        return True

api.add_resource(CreateBotHandler, '/api/ml/create_bot')
api.add_resource(ChatHandler, '/api/ml/chat')
api.add_resource(RetrainHandler, '/api/ml/retrain')
api.add_resource(TurnOnHandler, '/api/ml/turn_on')
api.add_resource(TurnOffHandler, '/api/ml/turn_off')
api.add_resource(UpdateParamsHandler, '/api/ml/update_params')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host=app.config['APP_ADDR'], port=app.config['APP_PORT'])
    # app.run(host=app.config['APP_ADDR'], port=app.config['APP_PORT'])
