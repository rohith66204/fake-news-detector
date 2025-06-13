# Create your views here.
from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import os
import pandas as pd

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def viewDataSet(request):
    path = os.path.join(settings.MEDIA_ROOT, 'FinalDataSet.csv' )
    df = pd.read_csv(path,  encoding = "ISO-8859-1")
    df = df[['source','user_name','FinalLabel','text','user_followers_count']]
    # print(df.head())
    df = df.to_html(index=None)
    return render(request, 'users/viewTweetDataset.html', {'data': df})

def userMachineLearning(request):
    from .utility import  mlprocessings
    cr_lg = mlprocessings.start_logisticRegression()
    cr_nb = mlprocessings.start_naivebayes()
    cr_svm = mlprocessings.start_svm()
    cr_rnn = mlprocessings.start_recurrentNeurals()
    return render(request, 'users/MlResults.html', {'cr_lg': cr_lg, 'cr_nb': cr_nb, 'cr_svm': cr_svm, 'cr_rnn': cr_rnn})


def start_test_predictions(request):
    if request.method=='POST':
        tweets = request.POST.get('tweets')
        import os
        import pandas as pd
        from django.conf import settings
        from sklearn.feature_extraction.text import CountVectorizer
        import pickle
        from sklearn.model_selection import train_test_split
        path1 = os.path.join(settings.MEDIA_ROOT, 'FinalDataSet.csv')
        path = os.path.join(settings.MEDIA_ROOT, 'fakenews.alex')
        df = pd.read_csv(path1, encoding="ISO-8859-1")
        df.shape

        df['FinalLabel'] = df.FinalLabel.map({'REAL': 1, 'FAKE': 0})
        X_train, X_test, y_train, y_test = train_test_split(df['text'],
                                                            df['FinalLabel'],
                                                            random_state=42)
        # Instantiate the CountVectorizer method
        count_vector = CountVectorizer(stop_words='english', lowercase=True)
        training_data = count_vector.fit_transform(X_train)
        # Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
        testing_data = count_vector.transform(X_test)
        test = count_vector.transform([tweets])
        model = pickle.load(open(path, 'rb'))
        pred = model.predict(test)
        if pred[0] == 1:
            msg = 'REAL'
        else:
            msg = 'FAKE'
        print("===>", pred)
        return render(request, 'users/test_pred.html', {'tweet': tweets, 'msg': msg})
    else:
        return render(request, 'users/test_pred.html', {})