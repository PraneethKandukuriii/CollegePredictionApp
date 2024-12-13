import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv('data/TS_EAMCET_2023_first_round-2.csv')


irrelevant_columns = ["Unnamed: 29", "Institute Name", "Place", "Dist Code", "Co\nEducati on", "College Type", "Year of Estab", "Tuition Fee", "TS\nAffiliated To"]
df_cleaned = df.drop(columns=irrelevant_columns)

encoder_branch = LabelEncoder()
encoder_college = LabelEncoder()

df_cleaned['Branch Name Encoded'] = encoder_branch.fit_transform(df_cleaned['Branch Name'])
df_cleaned['Inst Code Encoded'] = encoder_college.fit_transform(df_cleaned['Inst Code'])


df_cleaned['Combined Rank'] = df_cleaned.apply(lambda row: row["BC_B BOYS"], axis=1)
model_data = df_cleaned[['Combined Rank', 'Branch Name Encoded', 'Inst Code Encoded']].dropna()

X = model_data[['Combined Rank', 'Branch Name Encoded']]
y = model_data['Inst Code Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('encoder_branch.pkl', 'wb') as encoder_branch_file:
    pickle.dump(encoder_branch, encoder_branch_file)

with open('encoder_college.pkl', 'wb') as encoder_college_file:
    pickle.dump(encoder_college, encoder_college_file)

print("Model and encoders saved successfully!")
