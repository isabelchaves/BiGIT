**File descriptions**

* train.csv - the training set, contains products, searches, and relevance scores
* test.csv - the test set, contains products and searches. You must predict the relevance for these pairs.
product_descriptions.csv - contains a text description of each product. You may join this table to the training or test set via the product_uid.
* attributes.csv -  provides extended information about a subset of the products (typically representing detailed technical specifications). Not every product will have attributes.


**Data Fields**

* id - a unique Id field which represents a (search_term, product_uid) pair
* product_uid - an id for the products
* product_title - the product title
* product_description - the text description of the product (may contain HTML content)
* search_term - the search query
* relevance - the average of the relevance ratings for a given id
* name - an attribute name
* value - the attribute's value