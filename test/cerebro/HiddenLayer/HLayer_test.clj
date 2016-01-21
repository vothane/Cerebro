(ns cerebro.HiddenLayer.HLayer_test
  (:require [clojure.test :refer :all]
            [cerebro.HiddenLayer.HLayer :refer :all]))

(def hl [[ 0.06548973 -0.07128689 -0.09104952]
         [ 0.01710492  0.07315632 -0.02563118]
         [ 0.16025473  0.06160991 -0.00635603]
         [-0.03596083 -0.05227399  0.0763499 ]
         [-0.02047592 -0.14677403 -0.03398525]
         [ 0.0793318  -0.10583609 -0.10818275]])

(deftest hidden-layer-test
  (testing "hidden layer"
    (is (every? #{0 1} (flatten (sample-h-given-v hl [1 1 1 1 1 1] [1 1 1 0 0 0]))))))
