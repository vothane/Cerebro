(ns cerebro.LogisticRegression.LogReg_test
  (:require [clojure.test :refer :all]
            [cerebro.LogisticRegression.LogReg :refer :all]))

(deftest softmax-test
  (testing "softmax"
    (is (= (softmax [0.016666666666666666 -0.016666666666666666]) [0.5083325618141193 0.4916674381858807]))))

