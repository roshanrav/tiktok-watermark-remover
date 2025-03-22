import React, { useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { View, StyleSheet, ScrollView, Alert, Platform } from 'react-native';
import { Video } from 'expo-av';
import Constants from 'expo-constants';

import {
  Button,
  Provider as PaperProvider,
  Text,
  ActivityIndicator,
  Portal,
  Modal,
  List,
} from 'react-native-paper';
import * as DocumentPicker from 'expo-document-picker';
import * as Animatable from 'react-native-animatable';
import * as Font from 'expo-font';

// Navigation setup
const Stack = createStackNavigator();

// Backend URL from app.json
const API_URL = Constants.expoConfig.extra.API_URL;

function HomeScreen({ navigation }) {
  const [videoUri, setVideoUri] = useState(null);
  const [loading, setLoading] = useState(false);
  const [method, setMethod] = useState('blur');
  const [pickerVisible, setPickerVisible] = useState(false);

  // Method options: blur or inpaint
  const pickerOptions = [
    { label: 'Blur', value: 'blur' },
    { label: 'Inpaint', value: 'inpaint' },
  ];

  // Allow user to select an MP4 video
  const pickVideo = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: 'video/mp4',
        copyToCacheDirectory: true,
      });
      if (!res.canceled && res.assets?.length > 0) {
        setVideoUri(res.assets[0].uri);
      }
    } catch (err) {
      Alert.alert('Error picking file', err.message);
    }
  };

  // Send video and method to backend for processing
  const uploadAndProcess = async () => {
    if (!videoUri) {
      Alert.alert('No video selected', 'Please pick a video first.');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('video', {
        uri: videoUri,
        type: 'video/mp4',
        name: 'upload.mp4',
      });
      formData.append('method', method);

      const response = await fetch(`${API_URL}/process-video`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const text = await response.text();
      const result = JSON.parse(text);

      // Navigate to result screen with response data
      navigation.navigate('Result', {
        watermarkDetected: result.watermark_detected,
        processedVideoUri: result.processed_video_url,
      });
    } catch (err) {
      Alert.alert('Upload failed', err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Animatable.Text animation="bounceInDown" style={styles.logo}>
        🎬 CleanTok
      </Animatable.Text>
      <Text style={styles.subheader}>Your TikTok Watermark Remover</Text>

      <Button icon="video" mode="contained" onPress={pickVideo} style={styles.button}>
        Pick an MP4 Video
      </Button>

      {videoUri && (
        <>
          {/* Preview selected video */}
          <Animatable.View animation="fadeInUp" delay={300}>
            <Video
              source={{ uri: videoUri }}
              style={styles.video}
              useNativeControls
              resizeMode="contain"
            />
          </Animatable.View>

          {/* Picker for method selection */}
          <Animatable.View animation="fadeInUp" delay={400} style={styles.pickerContainer}>
            <Text style={styles.pickerLabel}>Choose Removal Method</Text>
            <Button
              mode="outlined"
              icon="chevron-down"
              onPress={() => setPickerVisible(true)}
              style={styles.dropdownButton}
              labelStyle={styles.dropdownButtonLabel}
            >
              {pickerOptions.find((opt) => opt.value === method)?.label || 'Choose'}
            </Button>

            <Portal>
              <Modal
                visible={pickerVisible}
                onDismiss={() => setPickerVisible(false)}
                contentContainerStyle={styles.modalContainer}
              >
                {pickerOptions.map((option) => (
                  <List.Item
                    key={option.value}
                    title={option.label}
                    onPress={() => {
                      setMethod(option.value);
                      setPickerVisible(false);
                    }}
                    left={(props) => (
                      <List.Icon
                        {...props}
                        icon={method === option.value ? 'check-circle' : 'circle-outline'}
                        color={method === option.value ? '#5e2bff' : '#aaa'}
                      />
                    )}
                    titleStyle={styles.modalItemText}
                  />
                ))}
              </Modal>
            </Portal>
          </Animatable.View>

          {/* Button to trigger upload */}
          <View style={styles.uploadButtonContainer}>
            <Button
              icon="cloud-upload"
              mode="contained"
              onPress={uploadAndProcess}
              style={styles.uploadButton}
              loading={loading}
              disabled={loading}
            >
              Upload & Process
            </Button>
          </View>
        </>
      )}

      {/* Show loader while uploading */}
      {loading && (
        <ActivityIndicator
          animating={true}
          size="large"
          color="#6200ee"
          style={{ marginTop: 30 }}
        />
      )}
    </ScrollView>
  );
}

function ResultScreen({ route, navigation }) {
  const { watermarkDetected, processedVideoUri } = route.params;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Button
        icon="arrow-left"
        mode="text"
        onPress={() => navigation.goBack()}
        style={styles.backButton}
      >
        Back
      </Button>

      <Animatable.Text animation="fadeInDown" style={styles.logo}>
        ✅ Results
      </Animatable.Text>

      {/* Show processed video if watermark was detected */}
      {watermarkDetected ? (
        <>
          <Animatable.Text animation="fadeIn" style={styles.detectedText}>
            Watermark Detected!
          </Animatable.Text>
          <Animatable.View animation="zoomIn" delay={300}>
            <Video
              source={{ uri: processedVideoUri }}
              style={styles.video}
              useNativeControls
              resizeMode="contain"
            />
          </Animatable.View>
        </>
      ) : (
        <Animatable.Text animation="fadeIn" delay={200} style={styles.noWatermarkText}>
          🎉 No Watermark Found!
        </Animatable.Text>
      )}
    </ScrollView>
  );
}

export default function App() {
  // Load fonts before rendering the app
  const [fontsLoaded] = Font.useFonts({
    'Poppins-Bold': require('./assets/fonts/Poppins-Bold.ttf'),
    'Poppins-Regular': require('./assets/fonts/Poppins-Regular.ttf'),
  });

  if (!fontsLoaded) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#6200ee" />
      </View>
    );
  }

  // Wrap app in PaperProvider and NavigationContainer
  return (
    <PaperProvider>
      <NavigationContainer>
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen name="Result" component={ResultScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}

// App styles
const styles = StyleSheet.create({
  container: {
    paddingTop: 80,
    paddingHorizontal: 20,
    alignItems: 'center',
    backgroundColor: '#f4f4f4',
    justifyContent: 'flex-start',
    minHeight: '100%',
  },
  logo: {
    fontSize: 36,
    fontFamily: 'Poppins-Bold',
    color: '#5e2bff',
    marginBottom: 10,
    textAlign: 'center',
  },
  subheader: {
    fontSize: 16,
    fontFamily: 'Poppins-Regular',
    color: '#555',
    marginBottom: 30,
    textAlign: 'center',
  },
  button: {
    marginTop: 20,
    width: '100%',
    borderRadius: 8,
  },
  video: {
    width: 330,
    height: 230,
    backgroundColor: 'black',
    borderRadius: 10,
    marginTop: 20,
  },
  pickerContainer: {
    width: '100%',
    marginTop: 20,
    alignItems: 'center',
  },
  pickerLabel: {
    fontFamily: 'Poppins-Bold',
    fontSize: 16,
    marginBottom: 8,
    color: '#5e2bff',
  },
  dropdownButton: {
    width: '80%',
    borderRadius: 5,
    borderColor: '#5e2bff',
    borderWidth: 1,
    marginTop: 8,
  },
  dropdownButtonLabel: {
    fontFamily: 'Poppins-Bold',
    fontSize: 16,
    color: '#5e2bff',
  },
  modalContainer: {
    backgroundColor: 'white',
    paddingVertical: 10,
    paddingHorizontal: 20,
    margin: 20,
    borderRadius: 12,
  },
  modalItemText: {
    fontFamily: 'Poppins-Regular',
    fontSize: 16,
    color: '#5e2bff',
  },
  uploadButtonContainer: {
    width: '100%',
    alignItems: 'center',
    marginTop: 20,
  },
  uploadButton: {
    width: '60%',
    borderRadius: 8,
  },
  backButton: {
    alignSelf: 'flex-start',
    marginBottom: 10,
  },
  detectedText: {
    fontSize: 20,
    color: 'green',
    fontFamily: 'Poppins-Bold',
    marginTop: 20,
    marginBottom: 10,
    textAlign: 'center',
  },
  noWatermarkText: {
    fontSize: 20,
    color: '#d32f2f',
    fontFamily: 'Poppins-Bold',
    marginTop: 40,
    textAlign: 'center',
  },
});
