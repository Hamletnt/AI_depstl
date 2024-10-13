import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import encoder as en
from fuzzywuzzy import process

# โหลดข้อมูลที่จำเป็น
df = pd.read_csv('cheese_encode_input.csv')
cheese_mapping_df = pd.read_csv('convert_cheese.csv')
loaded_rf_classifier = joblib.load('model_now.joblib')
loaded_predict_cheese_model = joblib.load('predict_cheese_model.joblib')

# Class Definitions
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stop criteria
        if (num_samples < self.min_samples_split) or (len(unique_classes) == 1) or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Best split
        best_feature, best_threshold = self._best_split(X, y)
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, X, y):
        num_features = X.shape[1]
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, feature_column, threshold):
        parent_entropy = self._entropy(y)
        left_indices = feature_column < threshold
        right_indices = feature_column >= threshold
        n = len(y)

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        child_entropy = (n_left / n * self._entropy(y[left_indices]) +
                         n_right / n * self._entropy(y[right_indices]))
        return parent_entropy - child_entropy

    def _entropy(self, y):
        class_labels, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict(sample, self.root) for sample in X])

    def _predict(self, sample, tree):
        if tree.value is not None:
            return tree.value
        if sample[tree.feature] < tree.threshold:
            return self._predict(sample, tree.left)
        else:
            return self._predict(sample, tree.right)

class RandomForestClassifierFromScratch:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            bootstrap_indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(tree_pred).argmax() for tree_pred in tree_preds.T])

# Function Definitions
def get_closest_match(value, valid_values):
    closest_match, score = process.extractOne(value, valid_values)
    return closest_match if score > 80 else None

def fill_missing_values(encoded_input, df):
    filled_input = []
    for i in range(len(encoded_input)):
        if encoded_input[i] is None:
            # Get the mode of the corresponding feature in the dataframe
            mode_value = df.iloc[:, i].mode().values[0]
            filled_input.append(mode_value)
        else:
            filled_input.append(encoded_input[i])
    return filled_input

def main():
    st.title("Cheese Prediction App")

    # Input fields for user to fill
    input_data = []
    index_key_mapping = ['milk', 'country', 'type', 'fat_content', 'texture', 'rind', 'color', 'flavor', 'aroma', 'vegetarian', 'vegan']
    
    for key in index_key_mapping:
        value = st.text_input(f"Enter value for {key}:")
        input_data.append(value if value else None)

    if st.button("Predict"):
        # เริ่มการแปลงค่า input โดยใช้ static_mapping และ fuzzy matching
        encoded_input = []
        
        for i, value in enumerate(input_data):
            col_name = index_key_mapping[i]
            if value is None:
                encoded_input.append(None)
            elif any(key.lower() == value.lower() and len(key) == len(value) for key in en.static_mapping[col_name].keys()):
                exact_key = next(key for key in en.static_mapping[col_name].keys() if key.lower() == value.lower() and len(key) == len(value))
                encoded_input.append(en.static_mapping[col_name][exact_key])
            else:
                closest_match = get_closest_match(value, en.static_mapping[col_name].keys())
                if closest_match:
                    encoded_input.append(en.static_mapping[col_name][closest_match])
                else:
                    encoded_input.append(None)

        # Convert to NumPy array
        encoded_input = np.array(encoded_input, dtype=object).reshape(1, -1)

        # เติมค่าที่ขาดหาย
        filled_input = fill_missing_values(encoded_input[0], df)

        # ใช้ model random forest ในการทำนาย
        predictions = loaded_rf_classifier.predict(np.array(filled_input, dtype=float).reshape(1, -1))

        # Convert predictions ให้เป็นชื่อ class
        class_names = ['Cheddar', 'Blue', 'Brie', 'Pecorino', 'Gouda', 'Parmesan', 'Camembert', 'Feta',
                       'Cottage', 'Pasta filata', 'Swiss Cheese', 'Mozzarella', 'Tomme']
        predictions_index = [3, 0, 1, 10, 6, 8, 2, 5, 4, 9, 11, 7, 12]
        predicted_class_names = [class_names[predictions_index.index(pred)] for pred in predictions]

        # แสดงผลลัพธ์
        st.success(f'Predicted Cheese: {predicted_class_names}')

        # การประมวลผลชีส (เช่นการใช้ loaded_predict_cheese_model)
        new_data = list(filled_input) + [predictions[0]]
        new_data_array = np.array(new_data, dtype=float).reshape(1, -1)

        feature_names = ['milk', 'country', 'family', 'type', 'fat_content', 
                         'texture', 'rind', 'color', 'flavor', 'aroma',
                         'vegetarian', 'vegan']
        
        data_df = pd.DataFrame(new_data_array, columns=feature_names)
        
        # Make predictions using the cheese model
        cheese_predictions = loaded_predict_cheese_model.predict(data_df)
        
        # Assuming column 1 has the class names and column 2 has the numeric values
        cheese_class_mapping = dict(zip(cheese_mapping_df.iloc[:, 1], cheese_mapping_df.iloc[:, 0]))

        # Convert numeric predictions to class names
        predicted_cheese_class_names = [cheese_class_mapping.get(pred, 'Unknown') for pred in cheese_predictions]

        # แสดงผลลัพธ์ชีส
        st.success(f'Cheese Classifications: {predicted_cheese_class_names}')

if __name__ == "__main__":
    main()
