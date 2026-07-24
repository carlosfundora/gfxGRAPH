# Assuming cudf acts as hipDF in the ROCm environment
try:
    import cudf
    import cupy as cp
except ImportError:
    print(
        "Warning: cudf or cupy not found. This script requires a ROCm-DS environment to execute."
    )


def run_pipeline():
    """
    Demonstrates a simple end-to-end pipeline integrating ROCm-DS components:
    1. hipDF (cudf) for data loading and ETL.
    2. CuPy for zero-copy data handoff.
    3. Mock hipRAFT/hipVS component for downstream processing.
    """
    print("Starting ROCm-DS end-to-end pipeline example...")

    # 1. ETL with hipDF
    print("Step 1: Data Loading with hipDF (cudf)...")
    try:
        # Create a sample dataframe directly on the GPU
        df = cudf.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "label": [0, 1, 0, 1, 0],
            }
        )
        print(f"Loaded DataFrame shape: {df.shape}")

        # Simple ETL operation
        df["combined_feature"] = df["feature1"] * df["feature2"]

        # 2. Zero-copy handoff using DLPack or CuPy
        print("Step 2: Zero-copy handoff to CuPy...")
        # Extract features for machine learning
        features_df = df[["feature1", "feature2", "combined_feature"]]

        # Convert to CuPy array without copying back to host memory
        features_cp = cp.from_dlpack(features_df.to_dlpack())
        print(f"CuPy array shape: {features_cp.shape}")

        # 3. Downstream Processing (e.g., hipRAFT clustering or hipVS indexing)
        print("Step 3: Downstream processing (Mocking hipRAFT)...")
        # In a real scenario, you would pass `features_cp` to a hipRAFT algorithm
        # Example: kmeans = hipraft.cluster.KMeans(n_clusters=2); kmeans.fit(features_cp)

        # Performing a simple CuPy operation to represent GPU work
        normalized_features = features_cp / cp.linalg.norm(features_cp, axis=0)
        print("Pipeline execution completed successfully. Data remained on GPU.")

    except Exception as e:
        print(f"Pipeline execution encountered an error: {e}")


if __name__ == "__main__":
    run_pipeline()
