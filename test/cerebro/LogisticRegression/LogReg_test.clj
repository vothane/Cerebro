(ns cerebro.LogisticRegression.LogReg_test
  (:require [clojure.test :refer :all]
            [cerebro.LogisticRegression.LogReg :refer :all]))

(deftest softmax-test
  (testing "softmax"
    (is (= (softmax [0.016666666666666666 -0.016666666666666666]) [0.5083325618141193 0.4916674381858807]))))

(deftest train-init-test
  (testing "train with initial state"
    (let [init-logreg {:weights [[0.0 0.0 0.0] [0.0 0.0 0.0]] :bias [0.0 0.0] :N 6}
          real-logreg {:weights [[0.008333333333333333 0.008333333333333333 0.0] [-0.008333333333333333 -0.008333333333333333 0.0]] 
                       :bias [0.008333333333333333 -0.008333333333333333]
                       :N 6}]                       
    (is (= (train init-logreg [1 1 0] [1 0] 0.1) real-logreg)))))

(deftest train-test
  (testing "train with one pass through of calculations"
    (let [init-logreg {:weights [[0.008333333333333333 0.008333333333333333 0.0] [-0.008333333333333333 -0.008333333333333333 0.0]] 
                       :bias [0.008333333333333333 -0.008333333333333333]
                       :N 6}
          real-logreg {:weights  [[0.016527790636431346 0.008333333333333333 0.0] [-0.016527790636431346 -0.008333333333333333 0.0]] 
                       :bias [0.016527790636431346 -0.016527790636431346]
                       :N 6}]                        
    (is (= (train init-logreg [1 0 0] [1 0] 0.1) real-logreg)))))