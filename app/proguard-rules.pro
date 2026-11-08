# ML Kit / TFLite 通过反射与 JNI 调用，保留其类
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_vision_pose** { *; }
-dontwarn com.google.mlkit.**

-keep class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**
