
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

import numpy
# Create a LabelEncoder
label_encoder = LabelEncoder()

# Encode a categorical feature
X =  numpy.array( ['red','blue','green'])
encoded_feature = label_encoder.fit_transform(X)

print(encoded_feature)

x_new = numpy.array( ['red'])
x_new_encoded =  label_encoder.transform(x_new)

print(x_new_encoded)

# Create a OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

print(X.reshape(-1,1))

# Transform a categorical feature into one-hot encoded columns
encoded_features = one_hot_encoder.fit_transform(X.reshape(-1,1))
print(encoded_features)
x_new_encoded = one_hot_encoder.transform(x_new.reshape(-1,1))
print(x_new_encoded)
