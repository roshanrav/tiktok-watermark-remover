import requests
## TEST FILE 



VIDEO_FILE = "test_video.mp4"
BACKEND_URL = "http://10.0.0.235:5001/process-video"

def test_watermark_removal(method):
    print(f"\n🔍 Testing method: {method}")

    with open(VIDEO_FILE, 'rb') as video:
        files = {
            'video': (VIDEO_FILE, video, 'video/mp4')
        }
        data = {
            'method': method
        }

        response = requests.post(BACKEND_URL, files=files, data=data)
        
        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            return

        result = response.json()

        print(f"✅ Processed: {result['processed_video_url']}")
        print(f"   - Watermark Detected: {result['watermark_detected']}")
        print(f"   - Count: {result['watermark_count']}")
        print(f"   - Debug Info: {result['debug_info']['id']}")

        assert 'processed_video_url' in result
        assert isinstance(result['watermark_detected'], bool)
        assert isinstance(result['watermark_count'], int)

test_watermark_removal('blur')
test_watermark_removal('inpaint')
