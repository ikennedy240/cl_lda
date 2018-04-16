"""
This module includes functions for supporting ML tasks and model comparision
"""


# Produces a score report with various output

print('AUC: ', roc_auc_score(y_test, predictions))
print('AUC: ', roc_auc_score(S_y, sub_predictions))
confusion_matrix(y_test, predictions)
confusion_matrix(S_y, sub_predictions)

accuracy_score(y_test, model.predict(vect.transform(X_test)))
f1_score(S_y, model.predict(vect.transform(S_X)))


# Returns the most important features

sorted_import_index = rf.feature_importances_.argsort()
sorted_import_index

print('Smallest Import:\n{}\n'.format(feature_names[sorted_import_index[:20]]))
print('Largest Import:\n{}\n'.format(feature_names[sorted_import_index[-400:]]))

feature_names = np.array(lf_vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = lf_model.coef_[0].argsort()
opp_coef_index = (-lf_model.coef_[0]).argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:99]]))
print('Largest Coefs: \n{}'.format(feature_names[opp_coef_index[:1000]]))
