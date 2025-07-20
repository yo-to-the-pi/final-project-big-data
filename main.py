import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Change this to your own student ID prefix
STUDENT_ID = "b11901093"

# List of (input_csv, output_csv_suffix)
DATASETS = [
    ("public_data.csv", f"{STUDENT_ID}_public.csv"),
    ("private_data.csv", f"{STUDENT_ID}_private.csv"),
]

def cluster_and_save(input_csv: str, output_csv: str):
    # 1. Load data
    df = pd.read_csv(input_csv)
    
    # 2. Extract features (drop 'id')
    X = df.drop(columns=["id"]).values
    
    # 3. Determine number of clusters from dimension
    D = X.shape[1]
    n_clusters = D * 4 - 1
    
    # 4. Build pipeline: scale → PCA (keep D dims) → KMeans
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=D, random_state=42),
        KMeans(
            n_clusters=n_clusters,
            n_init=50,
            max_iter=500,
            random_state=42
        )
    )
    
    # 5. Fit & predict
    labels = pipeline.fit_predict(X)
    
    # 6. Save submission
    submission = pd.DataFrame({
        "id": range(len(labels)),
        "label": labels
    })
    submission.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} (D={D}, clusters={n_clusters})")

def main():
    for inp, out in DATASETS:
        cluster_and_save(inp, out)

if __name__ == "__main__":
    main()


