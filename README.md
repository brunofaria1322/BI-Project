# BI-Project

Dataset:
https://www.kaggle.com/johndddddd/customer-satisfaction

*SECRET*
https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

Features that affects the satisfaction of customer
https://www.kaggle.com/fattahuzzaman/features-that-affects-the-satisfaction-of-customer

# Postgres Installation on Ubunto

```sh
# Create the file repository configuration:
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Import the repository signing key:
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# Update the package lists:
sudo apt-get update

# Install the latest version of PostgreSQL.
# If you want a specific version, use 'postgresql-12' or similar instead of 'postgresql':
sudo apt-get -y install postgresql
```

# More important features
```py
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf(n_estimators=100, random_state=0).fit(X,y),random_state=1).fit(X,y)
eli5.show_weights(perm, feature_names = X.columns.tolist())
```