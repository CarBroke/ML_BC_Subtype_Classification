import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn import svm, datasets
import hiplot as hip
from pySankey import sankey
from matplotlib.patches import Patch

st.title("METABRIC Breast Cancer Machine Learning App")
st.header('Created by Carson Broeker')
st.write("Breast cancer is a very heterogeneous disease. This means that every patient's breast cancer is somewhat unique. This manifests \
    in needing to be treated with different therapeutic options in the clinic. \
        However, because we cannot make a new drug for every single patient, we subsection patients into intrinsic subtypes which are likely to respond to a certain type of therapy. \
            Intrinsic subtypes tend to fall on a spectrum of prognostic outcomes, as shown below")

st.image("Intrinsic_Subtype_Prognosis.jpg")

st.write("One way to stratify these patients into intrinsic subtypes is by gene expression data. But there over 20,000 genes in the genome, \
    with most of these genes just being random noise and not affecting clinical outcomes or therapeutic decisions. There is a subset of 50 genes \
        called PAM50 that is FDA approved to stratify patients into each subtype. When you train a support vector machine learning model \
            on METABRIC patient gene expression data, it is very accurately able to predict what each additional patient's intrinsic subtype is as shown below.")

st.image("PAM50_ROC_Curve.png")

#st.write("Now, create your own machine learning model! First, decide what classifier system you want to use.")

df1 = pd.read_csv("data_clinical_patient.txt", sep="\t", skiprows=[1,2,3,4])
df1 = df1.dropna()
df = pd.read_csv("Downsampled_Metabric_Microarray.csv")
species = df.pop("CLAUDIN_SUBTYPE")
df = df.T


option_category = st.selectbox(
    'Play with each category to see how the different subtypes of breast cancer relate to other clinical aspects.',
    ['Cellularity',
 'Chemotherapy',
 'ER status measured by IHC',
 'HER2 status measured by SNP6',
 'Hormone Therapy',
 'Inferred Menopausal State',
 'Sex',
 'Integrative Cluster',
 'Overall Survival Status',
 'Pam50 + Claudin-low subtype',
 '3-Gene classifier subtype',
 "Patient's Vital Status",
 'Primary Tumor Laterality',
 'Radio Therapy',
 'Tumor Other Histologic Subtype',
 'Type of Breast Surgery',
 'Relapse Free Status'])

st.set_option('deprecation.showPyplotGlobalUse', False)

sankey_fig = sankey.sankey(
    df1["Pam50 + Claudin-low subtype"], df1[option_category], 
    aspect=20, fontsize=12, #figureName="Breast_Cancer"
)
st.pyplot(fig=sankey_fig)
plt.close()

random_number = st.slider("Now make your own machine learning model and see what parameters affect its accuracy. First, \
    choose the number of genes you want to train the model on.", min_value=1, max_value=250, value=50, step=1)

df = df.sample(n=random_number, random_state=0)
df = df.astype(dtype="float64")
df = df.T
lut = dict(zip(species.unique(), [0, 1, 2, 3, 4, 5, 6]))
numbers = species.map(lut)
y = np.array(numbers)
array = df.to_numpy().astype('int32')
array = array + abs(df.min().min())
normalized_array = stats.zscore(array, axis=None, nan_policy='omit')
X = normalized_array

test_fraction = st.slider("Next, select the test fraction size. The rest of the samples will be used to train the model.", min_value=0.1, max_value=0.9, value=0.2, step=0.1)
random_start = st.slider("Select a number to randomize the instantiation of the model.", min_value=0, max_value=100, step=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=random_start)

classifier_options = st.selectbox(
    "Next, decide what classifier system you want to use and test its accuracy.",
    [neighbors.KNeighborsClassifier(),
svm.SVC(kernel='rbf', probability=True, random_state=0),
GaussianProcessClassifier(),
DecisionTreeClassifier(),
RandomForestClassifier(),
MLPClassifier(max_iter=1000),
GaussianNB(),
AdaBoostClassifier(),
QuadraticDiscriminantAnalysis()])

agree = st.checkbox('If you need a hint on what is likely to be the best classifier, click here.')

if agree:
    st.write("Choose the SVC option.")

st.write("")

my_classifier = classifier_options.fit(X_train, y_train)
my_predictions = my_classifier.predict(X_test)
st.write("Accuracy is "+str(accuracy_score(y_test, my_predictions)))

st.write("Another way to visualize the sensitivity and specificity of the prediction is to use a receiver operating characteristic (ROC) curve. \
    It is a way to look at how good your model is from distinguishing the true group prediction from any of the other groups available \
        in a one vs all approach. We will use the support vector machine (SVM) classifier to instantiate the model on the number of random genes \
            you have chosen to train the model with. For area under the curve (AUC), the optimal value is 1 for each subtype, where 0.5 is no better \
                than randomly guessing which group if the sample belongs to its true group or any of the others.")

y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=random_start)

classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=0))

y_score = classifier.fit(X_train, y_train).decision_function(X_test)
print(classifier.score(X_test, y_test))

fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2
subtypes = ["LumA","Her2","claudin-low","Basal","LumB","Normal"]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(["blue","pink","yellow","red","#98F5FF","green"])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve for '+subtypes[i]+' (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Trained on '+str(random_number)+' Random Genes')
plt.legend(loc="lower right")
ax = plt.gca()
st.pyplot(ax.figure)
plt.close()

st.write("However, the previous example is supervised learning. What does the data look like when it doesn't have defined categories available? \
    This is where unsupervised hierarchical clustering comes into play. It simply clusters samples together based on how similar its features are. \
        If the genes selected have no predicting power, it will simply look like static and noise. But, if there is predictive power, \
            you will see large blocks of one color. Check how many samples of each subtype cluster together at the top as a gauge of predictive power! \
                Unlike supervised learning where weights for non-predictive features can be assigned to essentially zero, all features are weighted equally here.")

#species = df.pop("Pam50 + Claudin-low subtype")
lut = dict(zip(species.unique(), ["blue","pink", "yellow", "red", "#98F5FF", "green", "black"]))
row_colors = species.map(lut)
#newdf = newdf.astype('float64')
#df_z = newdf.apply(zscore)
#df_x = newdf.T
newdf = df.T

cluster_fig = sns.clustermap(newdf, vmin=-3, vmax=3, cmap='icefire', method='ward', z_score=0, \
    col_colors=row_colors, yticklabels=False, xticklabels=False)

handles = [Patch(facecolor=lut[name]) for name in lut]
plt.legend(handles, lut, title="Intrinsic Subtype", fontsize='large', \
    bbox_to_anchor=(1,1), bbox_transform=plt.gcf().transFigure, loc='upper right')

st.pyplot(cluster_fig)
plt.close()