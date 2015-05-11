(ns cerebro.LogisticRegression.LogReg_test
  (:require [clojure.test :refer :all]
            [cerebro.LogisticRegression.LogReg :refer :all]))

(deftest make-log-reg-test
  (testing "logistic regression"
    (let [lg (make-log-reg 2 3 4)]
      (is (= (:weights lg) [[0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0] [0.0 0.0 0.0 0.0]]))
      (is (= (:bias lg) '(0.0 0.0 0.0 0.0))))))

(deftest softmax-test
  (testing "softmax"
    (is (= (softmax '(1 2 3)) '(0.6652409557748218 1.3304819115496436 1.9957228673244656)))))

(deftest train-test
  (testing "train"
    (let [lg (make-log-reg 2 3 4)
          lg (assoc lg :weights [[1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0]])
          lg (assoc lg :bias '(2.0 2.0 2.0 2.0))
          lg (train lg '(1 2 3 4) '(4 5 6 7) 0.2)]  
    (is (= (:weights lg) [[1.4 1.5 1.6 1.7000000000000002] 
                          [1.8 2.0 2.2 2.4000000000000004] 
                          [2.2 2.5 2.8000000000000003 3.1000000000000005] 
                          [2.6 3.0 3.4000000000000004 3.8000000000000003]])))))