(ns cerebro.LogisticRegression.LogReg_test
  (:require [clojure.test :refer :all]
            [cerebro.LogisticRegression.LogReg :refer :all]))

(deftest make-log-reg-test
  (testing "logistic regression"
    (let [lg (make-log-reg 2 3 4)]
      (is (= (:weights lg) '((0.0 0.0 0.0 0.0) (0.0 0.0 0.0 0.0) (0.0 0.0 0.0 0.0))))
      (is (= (:bias lg) '(0.0 0.0 0.0 0.0))))))

(deftest softmax-test
  (testing "softmax"
    (is (= (softmax '(1 2 3)) '(0.6652409557748218 1.3304819115496436 1.9957228673244656)))))

(deftest train-test
  (testing "train"
    (let [lg (make-log-reg 2 3 4)
          lg (assoc lg :weights '((1.0 1.0 1.0 1.0) (1.0 1.0 1.0 1.0) (1.0 1.0 1.0 1.0)))
          lg (assoc lg :bias '(2.0 2.0 2.0 2.0))]  
    (is (= (train lg '(1 2 3 4) '(4 5 6 7) 0.2) '(0.6652409557748218 1.3304819115496436 1.9957228673244656))))))