# -*- coding: euc-kr -*- 

'''
Created on 2018. 4. 6.
sigmoid 
    binary Calssification 
      0 ~ 1 사이의 값을 가지도록함 ,  결과의 합이 1이 아님   
    
@author: ldh92
'''
import numpy as np

'''
softmax :
    Multiclass Classification  문제에서는 대표 적으로 사용 
    모든 클래스의 결과 값은 1 , 즉 클래스의 결과값이 활률과 같음 
     (
        sigmoid 함수와의 차이  전체 결과값을 고려하여 전체 합이 1이 되도록 함 ( 정규화)
        대상의 변화에의해 다른값의 weight 값이 영향을 받음 , 학습속도에 영향이 있음
     )
'''
def softmax( a ):
    c =np.max( a)
    exp_a = np.exp( a  -c)  # -c 보정처리 
    sum_exp_a = np.sum( exp_a)
    y =exp_a/sum_exp_a
    return y

def run_softmax():
    print("softmax:")
    input = np.array([1,1,1,1])
    print ( "입력 : " , input , "결과 :" , softmax( input) , "결과합 :" ,np.sum(softmax( input))) 
    #입력 :  [1 1 1 1] 결과 : [0.25 0.25 0.25 0.25] 결과합 : 1.0
    input = np.array([1,6,8,9])
    print ( "입력 : " , input , "결과 :" , softmax( input) , "결과합 :" ,np.sum(softmax( input)))
    #입력 :  [1 6 8 9] 결과 : [2.36574162e-04 3.51107187e-02 2.59435070e-01 7.05217637e-01] 결과합 : 1.0
    
def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum( t*np.log(y+delta))

def run_cross_entropy():
    y = np.array([0,0,0,1]) 
    y_ = np.array([0.1,0.21,0.3,0.7])
   
    print ( "예측이 맞을경우  , 입력 : " , y_ , "예측결과 :" ,y , " cost  :" ,cross_entropy(y,y_)) 
    #예측이 맞을경우  , 입력 :  [0.1  0.21 0.3  0.7 ] 예측결과 : [0 0 0 1]  cost  : 9.83203827708458
    y= np.array([1,0,0,0]) 
    y_ = np.array([0.1,0.21,0.3,0.7])
    print ( "예측이 틀릴경우  , 입력 : " , y_ , "예측결과 :" ,y , " cost  :" ,cross_entropy(y,y_))
    #예측이 틀릴경우  , 입력 :  [0.1  0.21 0.3  0.7 ] 예측결과 : [1 0 0 0]  cost  : 19.502895727659567 
    
    import tensorflow as tf
    
def run_nn():
    x = np.array([[1,2,10]])   
    #w = np.array([[1] , [2] , [2]])
    w = np.array([[1,2,3,4]
                       ,[1,2,3,4]
                       ,[1,2,3,4]])
    
    print("결과값 " , np.dot(x,w))
    print("softmax 0,0,0,1 확률" ,softmax( np.dot(x,w)) ,np.sum(softmax( np.dot(x,w)))) 
    print("[ cross_entropy 1,0,0,0 ] " , cross_entropy( np.array([1,0,0,0]), softmax( np.dot(x,w) )))
    print("[ cross_entropy 0,1,0,0 ] " ,cross_entropy( np.array([0,1,0,0]), softmax( np.dot(x,w) )))
    print("[ cross_entropy 0,0,1,0 ] " ,cross_entropy( np.array([0,0,1,0]), softmax( np.dot(x,w) )))
    print("[ cross_entropy 0,0,0,1 cost low ] " ,cross_entropy( np.array([0,0,0,1]), softmax( np.dot(x,w) )))
    # 결과값  [[13 26 39 52]]
    # softmax 0,0,0,1 확률 [[1.15481981e-17 5.10907748e-12 2.26032430e-06 9.99997740e-01]] 1.0000000000000002
    # [ cross_entropy 1,0,0,0 ]  16.11809565095832
    # [ cross_entropy 0,1,0,0 ]  16.118095650875972
    # [ cross_entropy 0,0,1,0 ]  16.118059218834862
    # [ cross_entropy 0,0,0,1 cost low ]  3.633220581536907e-05
    
    late =0.001
    gd = 1.5
    add_w = late*gd
    x = np.array([[1,2,10]])   
    #w = np.array([[1] , [2] , [2]])
    w = np.array([[1,2,3,4+add_w]
                       ,[1,2,3,4+add_w]
                       ,[1,2,3,4+add_w]])
    print("weight 변경 " )
    print("결과값 " , np.dot(x,w))
    print("softmax 0,0,0,1 확률" ,softmax( np.dot(x,w)) ,np.sum(softmax( np.dot(x,w)))) 
    print("[ cross_entropy 1,0,0,0 ] " , cross_entropy( np.array([1,0,0,0]), softmax( np.dot(x,w) )))
    print("[ cross_entropy 0,1,0,0 ] " ,cross_entropy( np.array([0,1,0,0]), softmax( np.dot(x,w) )))
    print("[ cross_entropy 0,0,1,0 ] " ,cross_entropy( np.array([0,0,1,0]), softmax( np.dot(x,w) )))
    print("[ cross_entropy 0,0,0,1 cost low ] " ,cross_entropy( np.array([0,0,0,1]), softmax( np.dot(x,w) )))

    # weight 변경 
    # 결과값  [[13.     26.     39.     52.0195]]
    # softmax 0,0,0,1 확률 [[1.13251901e-17 5.01041577e-12 2.21667504e-06 9.99997783e-01]] 1.0
    # [ cross_entropy 1,0,0,0 ]  16.11809565095832
    # [ cross_entropy 0,1,0,0 ]  16.11809565087756
    # [ cross_entropy 0,0,1,0 ]  16.118059922377853
    # [ cross_entropy 0,0,0,1 cost low ]  3.562866122999249e-05