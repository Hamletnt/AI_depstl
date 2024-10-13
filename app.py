from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import encoder as en
from fuzzywuzzy import process
from flask_cors import CORS

app = Flask(__name__)


# เปิดใช้งาน CORS สำหรับทุกเส้นทางในแอปพลิเคชัน (สามารถลบ CORS(app) นี้ได้ถ้าใช้ฟังก์ชันด้านบน)
CORS(app, origins="http://127.0.0.1:5500") 

df = pd.read_csv('cheese_encode_input.csv')
cheese_mapping_df = pd.read_csv('convert_cheese.csv')

# Function to fill NaN in the sample data based on nearest neighbors
def fill_missing_values(sample_data, df):
    # Get indices where there are no NaN values (only known values)
    known_columns = ~np.isnan(sample_data)

    # Extract only known values
    known_values = sample_data[known_columns]

    # Select the corresponding columns from the dataset that match known values
    columns_to_use = np.where(known_columns)[0]

    # Extract the relevant columns from the dataset for comparison
    df_relevant = df.iloc[:, columns_to_use]

    # Initialize the Nearest Neighbors model
    nearest_neighbors = NearestNeighbors(n_neighbors=1)

    # Fit the model on the relevant columns of the dataset
    nearest_neighbors.fit(df_relevant)

    # Find the closest row in the dataset to the known values
    _, indices = nearest_neighbors.kneighbors([known_values])

    # Get the closest row from the original dataset
    closest_row = df.iloc[indices[0][0]].values

    # Fill in the NaN values in the sample data with the closest row's values
    filled_sample = sample_data.copy()

    # Replace NaN values with the corresponding values from the closest row
    nan_columns = np.isnan(filled_sample)
    filled_sample[nan_columns] = closest_row[nan_columns]

    return filled_sample

# Decision Tree Node Class (Unchanged)
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        self.value = value

# Decision Tree Classifier (Unchanged)
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    
    def build_tree(self, dataset, curr_depth=0):
        ''' Recursive function to build the tree ''' 
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
        
            if best_split and best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                      left_subtree, right_subtree, best_split["info_gain"])

        # Create a leaf node if no further split is possible
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' Function to find the best split '''
    
        # Dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
    
        # Loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
        
            # Loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # Get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
            
                # Check if children are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                
                    # Compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                
                    # Update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "info_gain": curr_info_gain
                        }
                        max_info_gain = curr_info_gain

        # Return best split if found, otherwise return an empty dictionary
        return best_split if "info_gain" in best_split else None
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="gini"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# Random Forest Classifier Implementation
class RandomForestClassifierFromScratch:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, Y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, Y_sample = self._bootstrap_sampling(X, Y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, Y_sample)
            self.trees.append(tree)

    def _bootstrap_sampling(self, X, Y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], Y[indices]

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = [self._majority_vote(tree_predictions[:, i]) for i in range(X.shape[0])]
        return final_predictions

    def _majority_vote(self, predictions):
        return max(set(predictions), key=list(predictions).count)

# โหลดโมเดลที่ได้รับการฝึกมา
loaded_rf_classifier = joblib.load('model_now.joblib')

def get_closest_match(value, valid_values):
    # print(valid_values)
    if isinstance(value, str):  # ตรวจสอบว่าเป็น string หรือไม่
        # ตรวจสอบ exact match ก่อน
        if value in valid_values:
            return value
        
        # ตรวจสอบกรณีสลับตำแหน่งคำ เช่น 'United Kingdom, Scotland' กับ 'Scotland, United Kingdom'
        for valid_value in valid_values:
            if set(valid_value.split(', ')) == set(value.split(', ')):
                return valid_value
        
        # ใช้ fuzzy matching กับทุกค่าที่อยู่ใน valid_values
        matches = process.extract(value, valid_values, limit=None)
        
        # คัดเลือกค่าที่ดีที่สุด
        best_match = None
        best_score = 0

        for match, score in matches:
            #print(f"Matching {value} with candidate {match}, score: {score}")
            if score > best_score:
                best_score = score
                best_match = match
        
        # ปรับเกณฑ์คะแนนให้สูงขึ้นเพื่อความแม่นยำมากขึ้น
        return best_match if best_score >= 90 else None

    return None  # ถ้าไม่ใช่ string ให้คืนค่า None


def fill_missing_values(encoded_input, df):
    # Convert to NumPy array, handling None
    encoded_input = np.array([x if x is not None else np.nan for x in encoded_input], dtype=float)

    # Check for NaN values
    if np.any(np.isnan(encoded_input)):
        # Fill missing values using mode of the corresponding column
        for i in range(len(encoded_input)):
            if np.isnan(encoded_input[i]):
                # Get mode of the column from the original DataFrame
                mode_values = df.iloc[:, i].mode()  # Get mode of the column
                print(mode_values)
                if not mode_values.empty:  # Ensure there is at least one mode value
                    encoded_input[i] = mode_values[0]  # Use the first mode value
                    print(encoded_input)

    return encoded_input

# Declare the global variable to store encoded_input
shared_encoded_input = None
shared_prediction = None

# ปรับการคำนวณใน main function
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    global shared_encoded_input, shared_prediction

    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        headers = response.headers
        headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:5500'
        headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200

    try:
        # ดึงข้อมูล JSON จาก request
        data = request.get_json(force=True)
        input_data = data['input']  # ต้องเป็น list ของ string หรือ null
        print(input_data)

        # เริ่มการแปลงค่า input โดยใช้ static_mapping และ fuzzy matching
        encoded_input = []
        index_key_mapping = ['milk', 'country', 'type', 'fat_content', 'texture', 'rind', 'color', 'flavor', 'aroma', 'vegetarian', 'vegan']

        # ลูปที่แก้ไขการเช็คตามลำดับ index_key_mapping
        for i, value in enumerate(input_data[0]):
            if i < len(index_key_mapping):
                col_name = index_key_mapping[i]  # แทนที่การดึง col_name จาก static_mapping โดยใช้ index_key_mapping
                if value is None:  # ถ้าเป็นค่า null ให้เก็บค่า None
                    encoded_input.append(None)
                elif any(key.lower() == value.lower() and len(key) == len(value) for key in en.static_mapping[col_name].keys()):
                    exact_key = next(key for key in en.static_mapping[col_name].keys() if key.lower() == value.lower() and len(key) == len(value))
                    encoded_input.append(en.static_mapping[col_name][exact_key])
                else:  # ใช้ fuzzy matching หากไม่เจอ exact match
                    closest_match = get_closest_match(value, en.static_mapping[col_name].keys())
                    if closest_match:
                        encoded_input.append(en.static_mapping[col_name][closest_match])
                    else:
                        encoded_input.append(None)  # ถ้าไม่เจอ match ใดๆ ให้ใช้ None

        # ตรวจสอบ encoded_input
        print(encoded_input)
        shared_encoded_input = encoded_input

        # Convert to NumPy array
        encoded_input = np.array(encoded_input, dtype=object).reshape(1, -1)

        # เติมค่าที่ขาดหาย
        filled_input = fill_missing_values(encoded_input[0], df)

        # แปลง filled_input เป็น NumPy array
        filled_input = np.array(filled_input, dtype=float).reshape(1, -1)

        # ใช้ model random forest ในการทำนาย
        predictions = loaded_rf_classifier.predict(filled_input)
        shared_prediction = predictions
        print(predictions)

        # Convert predictions ให้เป็นชื่อ class
        class_names = ['Cheddar', 'Blue', 'Brie', 'Pecorino', 'Gouda', 'Parmesan', 'Camembert', 'Feta',
                       'Cottage', 'Pasta filata', 'Swiss Cheese', 'Mozzarella', 'Tomme']
        predictions_index = [3, 0, 1, 10, 6, 8, 2, 5, 4, 9, 11, 7, 12]
        predicted_class_names = [class_names[predictions_index.index(pred)] for pred in predictions]

        return jsonify({'prediction': predicted_class_names})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# โหลดโมเดลที่ได้รับการฝึกมา
loaded_predict_cheese_model = joblib.load('predict_cheese_model.joblib')

# Endpoint for cheese processing
@app.route('/cheese', methods=['POST', 'OPTIONS'])
def cheese():
    global shared_encoded_input, shared_prediction

    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        headers = response.headers
        headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:5500'
        headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200

    try:
        if shared_encoded_input is None:
            return jsonify({'error': 'No encoded input data found from predict().'}), 400

        # Use the shared prediction value directly in new_data
        new_data = shared_encoded_input[:]
        new_data.insert(2, shared_prediction[0])  # Insert prediction result at index 2
        print('New data array for cheese processing:', new_data)
        # Convert new_data to a NumPy array
        
        # Convert new_data to a NumPy array
        new_data_array = np.array(new_data, dtype=float).reshape(1, -1)

        feature_names = ['milk', 'country', 'family', 'type', 'fat_content', 
                 'texture', 'rind', 'color', 'flavor', 'aroma',
                 'vegetarian', 'vegan']  # เปลี่ยนชื่อฟีเจอร์ตามจริง
        
        data_df = pd.DataFrame(new_data_array,columns=feature_names)

        # Make predictions using the cheese model
        predictions = loaded_predict_cheese_model.predict(data_df)
        
         # Assuming column 1 has the class names and column 2 has the numeric values
        cheese_class_mapping = dict(zip(cheese_mapping_df.iloc[:, 1], cheese_mapping_df.iloc[:, 0]))

        # Convert numeric predictions to class names
        predicted_class_names = [cheese_class_mapping.get(pred, 'Unknown') for pred in predictions]

        # Return the predictions in JSON format
        return jsonify({'message': 'Success_cheese', 'data': predicted_class_names})

    except Exception as e:
        print('Error occurred in /cheese:', str(e))
        return jsonify({'error': str(e)}), 500
    
@app.route('/')
def hello():
    return "Hello World!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # เปิดให้เข้าถึงจากทุกอินเทอร์เฟซที่พอร์ต 5000
