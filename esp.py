#include <ESP8266WiFi.h>
#include <PubSubClient.h>

const char* ssid = "YOUR_WIFI";
const char* pass = "YOUR_PASS";

const char* mqttHost = "broker.hivemq.com";
const int   mqttPort = 1883;

// ต้องให้ตรงกับ Python
const char* topicAlert = "kllc/drowsy/demo001/alert";

WiFiClient espClient;
PubSubClient client(espClient);

const int LED_PIN = LED_BUILTIN;   // หรือกำหนดขาอื่น
const int BUZ_PIN = D5;            // ถ้ามี buzzer ต่อไว้

void callback(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i=0; i<length; i++) msg += (char)payload[i];

  if (String(topic) == topicAlert) {
    if (msg == "1") {
      digitalWrite(LED_PIN, LOW);   // LED_BUILTIN ติดเมื่อ LOW
      digitalWrite(BUZ_PIN, HIGH);
    } else {
      digitalWrite(LED_PIN, HIGH);
      digitalWrite(BUZ_PIN, LOW);
    }
  }
}

void reconnect() {
  while (!client.connected()) {
    String cid = "esp8266-" + String(ESP.getChipId(), HEX);
    if (client.connect(cid.c_str())) {
      client.subscribe(topicAlert);
    } else {
      delay(1000);
    }
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZ_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  digitalWrite(BUZ_PIN, LOW);

  Serial.begin(115200);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) delay(300);

  client.setServer(mqttHost, mqttPort);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();
}
