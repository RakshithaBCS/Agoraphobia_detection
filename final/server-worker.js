// Function to request notification permission
function requestNotificationPermission() {
    Notification.requestPermission(function(status) {
        console.log('Notification permission status:', status);
        if (status === 'granted') {
            // Schedule notification after 5 seconds
            setTimeout(showNotification, 5000);
        }
    });
}

// Function to show notification
function showNotification() {
    if (Notification.permission === 'granted') {
        var notificationOptions = {
            body: "Don't forget to visit our website!" // You can specify an icon here if needed
        };
        var notification = new Notification('Visit Our Website', notificationOptions);
    } else {
        console.log('Permission for notifications not granted.');
    }
}

// Event listener for notification button click
document.getElementById('notificationButton').addEventListener('click', requestNotificationPermission);
