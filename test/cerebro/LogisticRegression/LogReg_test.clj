(ns cerebro.LogisticRegression.LogReg_test
  (:require [clojure.test :refer :all]
            [cerebro.LogisticRegression.LogReg :refer :all]))

(deftest make-log-reg-test
  (testing "logistic regression"
    (let [lg (make-log-reg 6 3 4)]
      (is (= (:weights lg) [[0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0]]))
      (is (= (:bias lg) '(0.0 0.0 0.0 0.0))))))

(deftest softmax-test
  (testing "softmax"
    (is (= (softmax [1 2]) [0.26894142136999516 0.7310585786300049]))))

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
    (let [lg (make-log-reg 6 6 2)
          lg (train lg [1 1 1 0 0 0] [1 0] 0.01)]
    (is (= (:weights lg) [])))))

(deftest predict-test
  (testing "predict"
    (let [lg (make-log-reg 6 6 2)
          y  (predict lg [1 1 1 0 0 0] [1 0])]
    (is (= y [])))))