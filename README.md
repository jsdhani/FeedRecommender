
# Feed Recommendations

**NOTE: The detailed documentation of the proposed approach and the various recommendation use cases can be found in /documentation.pdf.**

**OVERVIEW**

The proposed approach uses a combination of unsupervised learning, supervised learning, and rule based approaches to create a feature set and train a multi-label classification model for finding the most suitable subset of contents for a given user.

The final recommendation set includes multiple recommendation types each accompanied with a fallback method to cover-up in case of failure in generating recommendations.

**PREPARING CONTENT PROFILE**

```python
/submission/content/
```

The scripts under this directory orchestrate the creation of content profile attributes that shall be utilised in the downstream tasks of unsupervised learning, i.e Clustering of contents.
The motivation behind clustering is to group contents on a criteria not solely based on their category and that also utilised their textual information. The entire pipeline involves multiple subcomponents, which during the process further add to the attribute set
(such as content source, text language etc.)

The content profile creation includes the following sub-procedures:

- Fetching Content Attributes
- Merging Content Attributes
- Processing Content Attributes
- Preparing Feature Set for Unsupervised Learning
- Clustering Contents

In order to execute the content profile generation procedures, use the [main.py](http://main.py) file at:

```python
/submission/content/main.py
```

In order to execute this script , use the following main():
```python
    if __name__ == '__main__':
        contentProfile.create_profile()
```

Any intermediate files/results are stored at

```python
/submission/content/intermediates/
```

**NOTE:** Make sure to download "en_core_web_lg" for spacy using the command
```python
python -m spacy download en_core_web_lg
```

**PREPARING USER PROFILE**

```python
/submission/user/
```
The scripts in this directory create user profile attributes to be used for downstream supervised neural model training. The attribute set comprises of 3 sub-components:

1) **User Info:** contain information about user's language preferences
2) **User Interaction:** contain information about each posts_id viewed by every user
3) **Content Info:** attributes particular to contents, including the obtained clustering results.

In order to execute the content profile generation procedures, use the [main.py](http://main.py) file at:

```python
/submission/user/main.py
```

Any intermediate files/results are stored at

```python
/submission/user/intermediates/
```

**ACCESSING THE MULTI-LABEL CLASSIFIER**

This multi-label classifier can be trained using the file

```python
/submission/user/model/train.py
```

The trained model is stored at

```python
/submission/user/model/01_userContentClusterClassifier
```

**GENERATING RECOMMENDATIONS**

```python
/submission/recsys/
```

The recommendation set comprises of 3 recommendation types:

- Recommendations based on user history
- Recommendations based on user Genre Preferences
- Trending Contents

The recommendation entry point is accessible at
```python
/submission/recsys/main.py
```

The entry point consolidates recommendations from all the recommendation types into a single recommendation set.

Each recommendation use case is accompanied by a fallback function, which can generate some recommendations if in case, the model fails to run its mainstream method.

### OBTAINING RECOMMENDATIONS

open your Terminal and go to the directory:

```python
/submission/recsys/
```

run the following command

```python
flask run
```

**Execution Screenshot**
![Alt text](img/execution.png?raw=true "Execution Screenshot")

