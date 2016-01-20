(ns cerebro.HiddenLayer.HLayer_test
  (:require [clojure.test :refer :all]
            [cerebro.HiddenLayer.HLayer :refer :all]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def w [[ 0.06548973 -0.07128689 -0.09104952]
        [ 0.01710492  0.07315632 -0.02563118]
        [ 0.16025473  0.06160991 -0.00635603]
        [-0.03596083 -0.05227399  0.0763499 ]
        [-0.02047592 -0.14677403 -0.03398525]
        [ 0.0793318  -0.10583609 -0.10818275]])

(deftest output-test
  (testing "output test"
    (let [hid-lay (make-hidden-layer 1 6 3)
          hid-lay (assoc hid-lay :weights w)]
      (is (= (output hid-lay [1 1 1 0 0 0] sigmoid) [0.0 0.0 0.0 0.0])))))