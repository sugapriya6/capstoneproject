# =========================
# 6. REAL-TIME PREDICTION WITH RISK CATEGORY
# =========================
y_test_prob = final_model.predict_proba(X_test)[:, 1]

# TEMPORARY - remove after checking
print("Min probability: ", y_test_prob.min())
print("Max probability: ", y_test_prob.max())
print("Mean probability:", y_test_prob.mean())
print("First 10 values: ", y_test_prob[:10])
