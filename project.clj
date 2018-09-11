(defproject mnist-classification "0.1.0-SNAPSHOT"
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 ; to manipulate images
                 [net.mikera/imagez "0.12.0"]
                 [thinktopic/cortex "0.9.22"]
                 [thinktopic/experiment "0.9.22"]
                 ]
  :uberjar-name "mnist.jar")
