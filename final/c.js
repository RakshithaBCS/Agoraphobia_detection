document.getElementById('sleepForm').addEventListener('submit', function(e) {
    e.preventDefault();
  
    const age = parseInt(document.getElementById('age').value);
    const bedtime = document.getElementById('bedtime').valueAsDate.getHours();
    const wakeup = document.getElementById('wakeup').valueAsDate.getHours();
  
    const recommendedSleepHours = age >= 18 ? 7 : 8;
  
    const sleepHours = (wakeup - bedtime + 24) % 24;
  
    let message;
    if (sleepHours >= recommendedSleepHours) {
      message = "You're getting enough sleep!";
    } else {
      message = "You should try to get more sleep.";
    }
  
    document.getElementById('result').innerHTML = `<p>${message}</p>`;
  });
  