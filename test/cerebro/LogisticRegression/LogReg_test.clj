(ns cerebro.LogisticRegression.LogReg_test
  (:require [clojure.test :refer :all]
            [cerebro.LogisticRegression.LogReg :refer :all]))

(deftest make-log-reg-test
  (testing "logistic regression"
    (let [lg (make-log-reg 6 3 4)]
      (is (= (:weights lg) [[0.0 0.0 0.0] [0.0 0.0 0.0] [0.0 0.0 0.0] [0.0 0.0 0.0]]))
      (is (= (:bias lg) [0.0 0.0 0.0 0.0])))))

(deftest softmax-test
  (testing "softmax"
    (is (= (softmax [1 2]) [0.26894142136999516 0.7310585786300049]))))

(deftest train-test
  (testing "train"
    (let [lg (make-log-reg 6 6 2)
          lg (train lg [1 1 1 0 0 0] [1 0] 0.1)]
    (is (= (:weights lg) [[ 0.008333333333333337  0.008333333333333337  0.008333333333333337 0.0 0.0 0.0]
                          [-0.008333333333333337 -0.008333333333333337 -0.008333333333333337 0.0 0.0 0.0]])))))

(deftest predict-test
  (testing "predict"
    (let [lg (make-log-reg 6 6 2)
          y  (predict lg [1 1 1 0 0 0])]
    (is (= y [0.5 0.5])))))

(deftest full-train-and-predict-test
  (testing "full training and prediction"
    (let [lg (make-log-reg 6 6 2)
           x [[1 1 1 0 0 0]
              [1 0 1 0 0 0]
              [1 1 1 0 0 0]
              [0 0 1 1 1 0]
              [0 0 1 1 0 0]
              [0 0 1 1 1 0]]
          y  [[1 0]
              [1 0]
              [1 0]
              [0 1]
              [0 1]
              [0 1]]
          xy (take 500 (cycle (map #(vector %1 %2) x y)))
          lg (reduce (fn [lg [x y]] (train lg x y 0.1)) lg xy)]
    (is (every? true? (map #(> 0.03 (Math/abs (- %1 %2)))
                           (predict lg [1 0 1 0 0 0])
                           [0.5109727550240598 0.4890272449759402])))
    (is (every? true? (map #(> 0.03 (Math/abs (- %1 %2)))
                           (predict lg [0 0 1 1 1 0])
                           [0.477601374630467 0.522398625369533]))))))

;0.5109727550240598 0.4890272449759402
;0.477601374630467 0.522398625369533