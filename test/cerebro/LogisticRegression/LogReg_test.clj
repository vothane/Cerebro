(ns cerebro.LogisticRegression.LogReg_test
  (:require [clojure.test :refer :all]
            [cerebro.LogisticRegression.LogReg :refer :all]))

(deftest log-reg-test
  (testing "logistic regression"
    (let [lg (make-log-reg 2 3 4)]
      (is (= (:weights lg) '((0.0 0.0 0.0 0.0) (0.0 0.0 0.0 0.0) (0.0 0.0 0.0 0.0))))
      (is (= (:bias lg) '(0.0 0.0 0.0 0.0))))))
