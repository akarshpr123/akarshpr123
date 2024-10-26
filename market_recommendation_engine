import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('dataset.csv')

# Prepare data for training
X = data[['ProductID']]  # Features
y = data['Rating']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)


# Function to recommend products for a specific customer
def recommend_products(customer_id):
    # Get all product IDs
    product_ids = data['ProductID'].unique()

    # Create a DataFrame for predictions
    product_ids_df = pd.DataFrame(product_ids, columns=['ProductID'])

    # Predict ratings for all products
    predicted_ratings = model.predict(product_ids_df)

    # Combine predictions with product IDs
    recommendations = pd.DataFrame({
        'ProductID': product_ids,
        'PredictedRating': predicted_ratings
    })

    # Sort by predicted rating
    recommendations = recommendations.sort_values(by='PredictedRating', ascending=False)

    # Get the top recommended products
    top_recommendations = recommendations.head(5)

    return top_recommendations


# Function to get product name
def get_product_name(product_id):
    product = data[data['ProductID'] == product_id]
    return product['ProductName'].values[0] if not product.empty else "Unknown Product"


# Example usage
if __name__ == "__main__":
    customer_to_recommend = 1  # Example customer ID
    recommended_products = recommend_products(customer_to_recommend)

    print(f"Top recommendations for Customer {customer_to_recommend}:")
    for _, row in recommended_products.iterrows():
        product_name = get_product_name(row['ProductID'])
        print(f"Product: {product_name}, Predicted Rating: {row['PredictedRating']:.2f}")
