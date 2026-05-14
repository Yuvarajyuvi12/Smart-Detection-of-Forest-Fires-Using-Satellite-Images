
/**************** 🔥 FIRE ****************/
const fireForm = document.getElementById("fire-form");
const fireImage = document.getElementById("fire-image");
const firePreview = document.getElementById("fire-preview");
const fireResult = document.getElementById("fire-result");
const fireFileName = document.getElementById("fire-file-name");

fireImage.addEventListener("change", () => {
    const file = fireImage.files[0];
    if (file) {
        fireFileName.textContent = file.name;
        firePreview.src = URL.createObjectURL(file);
        firePreview.style.display = "block";
    }
});

function animateBar(bar, target) {
    let width = 0;
    const interval = setInterval(() => {
        if (width >= target) {
            clearInterval(interval);
        } else {
            width++;
            bar.style.width = width + "%";
            bar.innerText = width + "%";
        }
    }, 10);
}

fireForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!fireImage.files.length) {
        fireResult.innerHTML = "<p class='error'>❌ No file selected</p>";
        return;
    }

    fireResult.innerHTML = "<p class='loading'>⏳ Detecting...</p>";

    const formData = new FormData();
    formData.append("image", fireImage.files[0]);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if (data.prediction) {
        let html = `<h3>Prediction: <span class="prediction-label">${data.prediction}</span></h3>`;

        for (const [label, prob] of Object.entries(data.probabilities)) {
            const percent = Math.round(prob * 100);
            let color = "#4CAF50";
            if (label === "Fire") color = "#ff3b3b";
            if (label === "Smoke") color = "#ffaa00";

            html += `
                <div class="bar-wrapper">
                    <div class="bar-label">${label}</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="background:${color}; width:0%" data-target="${percent}"></div>
                    </div>
                </div>
            `;
        }

        fireResult.innerHTML = html;

        document.querySelectorAll(".bar-fill").forEach(bar => {
            animateBar(bar, bar.dataset.target);
        });
    }
});


/**************** 🍃 LEAF ****************/
const leafForm = document.getElementById("leaf-form");
const leafImage = document.getElementById("leaf-image");
const leafPreview = document.getElementById("leaf-preview");
const leafResult = document.getElementById("leaf-result");
const leafFileName = document.getElementById("leaf-file-name");

leafImage.addEventListener("change", () => {
    const file = leafImage.files[0];
    if (file) {
        leafFileName.textContent = file.name;
        leafPreview.src = URL.createObjectURL(file);
        leafPreview.style.display = "block";
    }
});

leafForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!leafImage.files.length) {
        leafResult.innerHTML = "<p class='error'>❌ No file selected</p>";
        return;
    }

    leafResult.innerHTML = "<p class='loading'>⏳ Analyzing leaf...</p>";

    const formData = new FormData();
    formData.append("leaf_image", leafImage.files[0]);

    const res = await fetch("/predict_leaf", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if (data.prediction) {
        leafResult.innerHTML = `
            <h3>Leaf Condition:
                <span style="color:${data.prediction === 'Green Leaf' ? 'green' : 'brown'}">
                    ${data.prediction}
                </span>
            </h3>
        `;
    }
});


















