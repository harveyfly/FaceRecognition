const video = document.getElementById('camera');
const canvas = document.getElementById('canvas');
const button = document.getElementById('getVideo');
let currentStream;

function stopMediaTracks(stream) {
  stream.getTracks().forEach(track => {
    track.stop();
  });
}

function gotDevices(mediaDevices) {
  $("#select").innerHTML = '';
  $("#select").append(document.createElement('option'));
  let count = 1;
  mediaDevices.forEach(mediaDevice => {
    if (mediaDevice.kind === 'videoinput') {
      var label = mediaDevice.label || `Camera ${count++}`;
      var value = mediaDevice.deviceId;
      var option = "<options value='"+value+"'>"+label+"</option>";
      $("#select").append(option);
      $("#select").selectpicker('refresh');
    }
  });
}

button.addEventListener('click', event => {
  if (typeof currentStream !== 'undefined') {
    stopMediaTracks(currentStream);
  }
  const videoConstraints = {};
  if ($("#select").value === '') {
    videoConstraints.facingMode = 'environment';
  } else {
    videoConstraints.deviceId = { exact: $("#select").value };
  }
  const constraints = {
    video: videoConstraints,
    audio: false
  };
  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(stream => {
      currentStream = stream;
      video.srcObject = stream;
      return navigator.mediaDevices.enumerateDevices();
    })
    .then(gotDevices)
    .catch(error => {
      console.error(error);
    });
});

navigator.mediaDevices.enumerateDevices().then(gotDevices);

// 触发拍照动作
document.getElementById("snap").addEventListener("click", function() {
  var context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, 640, 480);
});
