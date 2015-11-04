(ns cerebro.LogisticRegression.LogReg_test
  (:require [clojure.test :refer :all]
            [cerebro.LogisticRegression.LogReg :refer :all]))

(deftest make-log-reg-test
  (testing "logistic regression"
    (let [lg (make-log-reg 3 4)]
      (is (= (:weights lg) [[0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0]]))
      (is (= (:bias lg) '(0.0 0.0 0.0 0.0))))))

(deftest softmax-test
  (testing "softmax"
    (is (= (softmax [[1 2 3] [1 2 3]]) 
           [[0.045015286585190224 0.1223642355273988 0.3326204778874109]
            [0.045015286585190224 0.1223642355273988 0.3326204778874109]]))))

(def x [[1 1 1 0 0 0]
        [1 0 1 0 0 0]
        [1 1 1 0 0 0]
        [0 0 1 1 1 0]
        [0 0 1 1 0 0]
        [0 0 1 1 1 0]])

(def y [[1 0]
        [1 0]
        [1 0]
        [0 1]
        [0 1]
        [0 1]])

(deftest train-test
  (testing "train"
    (let [lg (make-log-reg 6 2)
          lg (train lg x y 0.01)]
    (is (= (:weights lg) [[ 0.015 -0.015]
                          [ 0.01  -0.01 ]
                          [ 0.0    0.0  ]
                          [-0.015  0.015]
                          [-0.01   0.01 ]
                          [ 0.0    0.0  ]])))))