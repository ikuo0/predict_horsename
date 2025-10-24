
function log_info() {
    console.log.apply(null, arguments);
}

function log_error() {
    console.error.apply(null, arguments);
}

function Array1D() {
    var self = this;
    self.zeros = function(length) {
        return new Array(length).fill(0);
    }

    self.max = function(arr) {
        return Math.max.apply(null, arr);
    }

    self.linalg_norm = function(arr) {
        var sum = 0;
        for (var i = 0; i < arr.length; i++) {
            sum += arr[i] * arr[i];
        }
        return Math.sqrt(sum);
    }

    self.sum_1d = function(arr) {
        return arr.reduce(function(a, b) { return a + b; }, 0);
    }
    self.div_1d_scalar = function(arr, scalar) {
        return arr.map(function(x) { return x / scalar; });
    }
    self.power_1d_scalar = function(arr, power) {
        return arr.map(function(x) { return Math.pow(x, power); });
    }

    self.where_greater_indexes = function(arr, threshold) {
        var indexes = [];
        for (var i = 0; i < arr.length; i++) {
            if (arr[i] > threshold) {
                indexes.push(i);
            }
        }
        return indexes;
    }
    self.argsort = function(arr) {
        var indexes = Array.from(arr.keys());
        indexes.sort(function(a, b) { return arr[b] - arr[a]; });
        return indexes;
    }
    self.indexing = function(arr, indexes) {
        return indexes.map(function(i) { return arr[i]; });
    }

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // 正規化関数
    ////////////////////////////////////////////////////////////
    // L2正規化
    // 配列、最小値保証
    self.l2_normalize = function(arr) {
        var me = this;
        var norm = self.linalg_norm(arr);
        norm = me.max([norm, 1e-10]);
        return self.div_1d_scalar(arr, norm);
    }
}

var n1d = new Array1D();

function UtilsFunction() {
    var self = this;
    self.ajax_get_json = async function(url, resolve, reject) {
        try {
            log_info("AJAX GET JSON: " + url);
            var response = await $.ajax({
                url: url,
                method: "GET",
                dataType: "json",
                cache: false,
                timeout: 10000
            });
            resolve(response);
        } catch(error) {
            log_error("AJAX GET JSON ERROR: " + url, error);
            reject(error);
        }
    }

}
var utils = new UtilsFunction();

function FeatureFunction() {
    var self = this;
    self.characters_list = null;
    self.char_to_index = {};

    self.load_characters_list = function() {
        var me = this;
        return new Promise(function(resolve, reject) {
            utils.ajax_get_json(
                "./model/futurize_characters_summary.json",
                function(response) {
                    me.characters_list = response["chars"];
                    for (var i = 0; i < me.characters_list.length; i++) {
                        var ch = me.characters_list[i];
                        me.char_to_index[ch] = i;
                    }
                    log_info("Loaded characters list. Length: " + me.characters_list.length);
                    resolve(true);
                },
                function(error) {
                    log_error("Failed to load characters list.", error);
                    reject(error);
                }
            );
        });
    }

    self.name_part_to_feature = function(char_list) {
        var me = this;
        var feat = n1d.zeros(me.characters_list.length);
        var size = char_list.length;
        for (var i = 0; i < char_list.length; i++) {
            var ch = char_list[i];
            if (ch in me.char_to_index) {
                var idx = me.char_to_index[ch];
                var weight = 1/ (size - i);
                feat[idx] += weight;
            } else {
                log_error("Character not in index: [" + ch + "]");
            }
        }
        return n1d.l2_normalize(feat);
    }
    self.get_horse_name_predict_prefix = function(horse_name) {
        var PART_SIZE = 4;
        var me = this;
        // 文字列を文字配列に変換
        var horse_chars = horse_name.split("");
        // "BOS" * PART_SIZE で前を埋める
        var padded_char_list = new Array(PART_SIZE).fill("BOS_PAD").concat(horse_chars);
        // 末尾 PART_SIZE 個の配列を作成
        var part = padded_char_list.slice(-PART_SIZE);
        return part;
    }
    self.get_horse_name_predict_feat = function(horse_name) {
        var me = this;
        var part = self.get_horse_name_predict_prefix(horse_name);
        var feat = me.name_part_to_feature(part);
        return feat;
    }
    self.setup = function() {
        var me = this;
        me.load_characters_list();
    }

    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // predict functions
    ////////////////////////////////////////////////////////////
    self.power_smoothing = function(prob_arr, power) {
        var me = this;
        var powered = n1d.power_1d_scalar(prob_arr, power);
        var sum_powered = n1d.sum_1d(powered);
        if (sum_powered < 1e-10) {
            return n1d.div_1d_scalar(powered, 1e-10);
        } else {
            return n1d.div_1d_scalar(powered, sum_powered);
        }
    }
}

function TensorFlowFunction() {
    var self = this;
    log_info("TensorFlow.js version:", tf.version.tfjs);
    self.model = null;
    self.setup = function() {
        var me = this;
        return new Promise(function(resolve, reject) {
            // tf.loadLayersModel("./model/model.json")
            tf.loadGraphModel("./model/model.json")
            .then(function(model) {
                me.model = model;
                log_info("Loaded TensorFlow model.");
                resolve(true);
            });
        });
    }
    self.dispose = function() {
        var me = this;
        if (me.model) {
            me.model.dispose();
            me.model = null;
        }
    }
    self.predict = async function(feature) {
        var me = this;
        return new Promise(function(resolve, reject) {
            if (!me.model) {
                reject("Model not loaded.");
                return;
            }
            var input = tf.tensor2d([feature]);
            var output = me.model.predict(input);
            output.array().then(function(outputData) {
                var row0 = outputData[0];
                input.dispose();
                output.dispose();
                resolve(row0);
            });
        });
        // if (!me.model) {
        //     await me.setup();
        // }
        // var input = tf.tensor2d([feature]);
        // var output = me.model.predict(input);
        // var outputData = await output.data();
        // input.dispose();
        // output.dispose();
        // var row0 = outputData[0];
        // return row0;
    }
}

function InputValues() {
    var self = this;
    self.prefix_string = "";
    self.smooth_coefficient = 0.8;
    self.predict_count = 10;

    self.validation = function() {
        var me = this;
        // prefix_string は空でも良い、カタカナのみ（濁点、半濁点、伸ばし棒もOK）
        var katakanaRegex = /^[\u30A0-\u30FF]+$/;
        if (me.prefix_string && !katakanaRegex.test(me.prefix_string)) {
            alert("プレフィックスはカタカナのみで入力してください。");
            return false;
        }
        // smooth_coefficient は 0.1 ~ 2.0 の範囲
        if (isNaN(me.smooth_coefficient) || me.smooth_coefficient < 0.1 || me.smooth_coefficient > 2.0) {
            log_error("smooth_coefficient:", me.smooth_coefficient);
            alert("スムース係数は0.1から2.0の範囲で入力してください。");
            return false;
        }
        // predict_count は 1 ~ 50 の範囲
        if (isNaN(me.predict_count) || me.predict_count < 1 || me.predict_count > 50) {
            alert("予測数は1から50の範囲で入力してください。");
            return false;
        }
        return true;
    }

    self.load_from_ui = function() {
        var me = this;
        me.prefix_string = $("#inputPrefixString").val();
        me.smooth_coefficient = parseFloat($("#inputSmoothCoefficient").val());
        me.predict_count = parseInt($("#inputPredictCount").val());
        me.validation();
    }
}

function IndexMain() {
    var self = this;
    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////
    // predict functions
    ////////////////////////////////////////////////////////////
    //
    self._execute_predict = async function(prefix_string, smooth_coeffient) {
        var me = this;
        log_info("Execute predict:");
        log_info("  prefix_string:", prefix_string);
        log_info("  smooth_coefficient:", smooth_coeffient);

        var horse_name = prefix_string;
        var HORSE_NAME_SIZE_MAX = 12;
        var size = horse_name.length;
        while(size < HORSE_NAME_SIZE_MAX) {
            var next_char_feat = await me.featureFunction.get_horse_name_predict_feat(horse_name);
            var row0 = await me.tensorFlowFunction.predict(next_char_feat);
            var smoothed_feat = me.featureFunction.power_smoothing(row0, smooth_coeffient);
            var char_index = me.get_next_char_index(smoothed_feat, 5);
            var next_char = me.featureFunction.characters_list[char_index];
            log_info("horse_name: ", horse_name);
            if (next_char === "EOS_PAD") {
                if(horse_name.length > prefix_string.length && horse_name.length > 2) {
                    break;
                } else {
                    // 文字数が足りてない場合は継続
                    continue;
                }
            }
            horse_name += next_char;
            size += 1;
        }
        log_info("Final horse name: ", horse_name);
        return horse_name;
    }
    self.get_input_values = function() {
        var me = this;
        var inputValues = new InputValues();
        inputValues.load_from_ui();
        return inputValues;
    }
    self.execute_predict = async function() {
        var me = this;
        var predict_names = [];
        var inputValues = me.get_input_values();
        for (var i = 0; i < inputValues.predict_count; i++) {
            var predicted_name = await me._execute_predict(
                inputValues.prefix_string,
                inputValues.smooth_coefficient
            );
            predict_names.push(predicted_name);
        }
        log_info("Predicted names:", predict_names);
        return predict_names;
    }
    self.result_display = function(horse_names) {
        var me = this;
        var $resultTextArea = $("#predictionResult");
        var joinNames = horse_names.join("\n");
        $resultTextArea.empty();
        $resultTextArea.val(joinNames);
    }

    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////
    // setup functions
    /////////////////////////////////////////////////////////////
    self.setup_events = function() {
        var me = this;
        // イベントのセットアップ
        // 予測ボタン押下
        $("#predictButton").on("click", function() {
            // 予測処理を実行
            me.execute_predict()
            .then(function(horse_names) {
                me.result_display(horse_names);
                return true;
            });
        });
        return Promise.resolve(true);
    }
    self.setup = function() {
        var me = this;
        return me.setup_events()
        .then(function() {
            log_info("Events set up.");
            me.featureFunction = new FeatureFunction();
            return me.featureFunction.setup();
        })
        .then(function() {
            log_info("FeatureFunction set up.");
            me.tensorFlowFunction = new TensorFlowFunction();
            return me.tensorFlowFunction.setup();
        })
        .then(function() {
            log_info("TensorFlowFunction set up.");
            return true;
        });
    }

    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////
    // predict functions
    /////////////////////////////////////////////////////////////
    self.get_top5_indexes = function(prob_arr) {
        var me = this;
        log_error("prob_arr", prob_arr)
        var indexes = n1d.argsort(prob_arr);
        return indexes.slice(0, 5);
    }
    self.probability_random_select = function(prob_arr) {
        var me = this;
        var prob_sum = n1d.sum_1d(prob_arr);
        var r = Math.random() * prob_sum;
        var cumulative = 0.0;
        for(var i = 0; i < prob_arr.length; i++) {
            cumulative += prob_arr[i];
            if (r < cumulative) {
                return i;
            }
        }
        return prob_arr.length - 1;
    }
    self.get_next_char_index = function(prob_arr, top_k) {
        var me = this;
        var indexes = n1d.argsort(prob_arr);
        var top_indexes = indexes.slice(0, top_k);
        var top_probs = n1d.indexing(prob_arr, top_indexes);
        var selected_index = me.probability_random_select(top_probs);
        return top_indexes[selected_index];
    }
    
    self.test_predict = async function() {
        var me = this;
        var horse_name_prefix = "イクノ";
        //var feat = me.featureFunction.get_horse_name_predict_feat(horse_name_prefix);

        var HORSE_NAME_SIZE_MAX = 12;
        var size = horse_name_prefix.length;
        while(size < HORSE_NAME_SIZE_MAX) {
            var next_char_feat = await me.featureFunction.get_horse_name_predict_feat(horse_name_prefix);
            // var smoothed_feat = me.featureFunction.power_smoothing(next_char_feat, 0.8);
            var row0 = await me.tensorFlowFunction.predict(next_char_feat);
            var smoothed_feat = me.featureFunction.power_smoothing(row0, 0.8);
            var char_index = me.get_next_char_index(smoothed_feat, 5);
            var next_char = me.featureFunction.characters_list[char_index];
            log_info("horse_name_prefix: ", horse_name_prefix);
            if (next_char === "EOS_PAD") {
                break;
            }
            horse_name_prefix += next_char;
            size += 1;
        }
        log_info("Final horse name: ", horse_name_prefix);
    }
    self.main = function() {
        var me = this;
        console.log("IndexMain");
        me.setup()
        .then(function() {
            log_info("Setup complete.");
            // me.test_predict();
            return true;
        });
    }
}
