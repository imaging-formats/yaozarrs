function createYaozarrsAnimation() {
    const yaozarrsElements = document.querySelectorAll('.yaozarrs-animated');

    yaozarrsElements.forEach(element => {
        const text = element.textContent.trim();
        element.innerHTML = '';

        // Create individual letter spans
        const surpriseColors = [
            '#ff4081', '#e91e63', '#9c27b0', '#673ab7',
            '#3f51b5', '#2196f3', '#00bcd4', '#009688',
            '#4caf50', '#8bc34a', '#ffeb3b', '#ff9800', '#ff5722'
        ];

        const letters = text.split('').map((letter, index) => {
            const span = document.createElement('span');
            span.textContent = letter === ' ' ? '\u00A0' : letter;
            span.className = 'yaozarrs-letter';

            // Assign a vibrant color to each letter
            const color = surpriseColors[index % surpriseColors.length];

            span.style.cssText = `
                display: inline-block;
                opacity: 0;
                color: ${color};
                transform: scale(0) translateY(20px) rotate(-10deg);
                transition: transform 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275), opacity 0.5s ease;
                transition-delay: ${index * 0.06}s;
            `;

            // Store original position after initial animation
            span.originalPosition = null;

            return span;
        });

        letters.forEach(letter => element.appendChild(letter));

        // Function to update original positions
        const updateOriginalPositions = () => {
            const rects = letters.map(letter => letter.getBoundingClientRect());
            rects.forEach((rect, i) => {
                letters[i].originalPosition = {
                    x: rect.left + rect.width / 2,
                    y: rect.top + rect.height / 2
                };
            });
        };

        // Trigger animation on load
        setTimeout(() => {
            letters.forEach(letter => {
                letter.style.opacity = '1';
                letter.style.transform = 'scale(1) translateY(0px) rotate(0deg)';
            });

            // Store original positions after animation settles
            setTimeout(updateOriginalPositions, 600);
        }, 100);

        // Update positions on window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(updateOriginalPositions, 100);
        });

        // Add interactive mouse-following effect
        element.addEventListener('mousemove', (e) => {
            const mouseX = e.clientX;
            const mouseY = e.clientY;

            letters.forEach(letter => {
                // Skip if original position not yet stored
                if (!letter.originalPosition) return;

                // Use original position instead of current position
                const letterCenterX = letter.originalPosition.x;
                const letterCenterY = letter.originalPosition.y;

                // Calculate distance from mouse to letter's original position
                const distanceX = mouseX - letterCenterX;
                const distanceY = mouseY - letterCenterY;
                const distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);

                // Create magnetic effect - stronger when closer
                const maxDistance = 150;
                const influence = Math.max(0, (maxDistance - distance) / maxDistance);

                // Calculate repulsion effect (letters move away from mouse)
                const pushX = distanceX > 0 ? -influence * 15 : influence * 15;
                const pushY = distanceY > 0 ? -influence * 15 : influence * 15;

                // Add some rotation and scale based on mouse proximity
                const rotation = influence * (distanceX > 0 ? 15 : -15);
                const scale = 1 + influence * 0.3;

                letter.style.transform = `
                    scale(${scale})
                    translateY(${pushY}px)
                    translateX(${pushX}px)
                    rotate(${rotation}deg)
                `;
                letter.style.transition = 'all 0.15s ease-out';
            });
        });

        element.addEventListener('mouseleave', () => {
            letters.forEach(letter => {
                letter.style.transform = 'scale(1) translateY(0px) translateX(0px) rotate(0deg)';
                letter.style.transition = 'all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
            });
        });
    });
}

// Add CSS styles
const style = document.createElement('style');
style.textContent = `
    .yaozarrs-animated {
        font-weight: bold;
        font-size: 4em;
        cursor: pointer;
        user-select: none;
        display: block;
        text-align: center;
        margin: 20px 0;
    }

    .yaozarrs-letter {
        text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
    }
`;
document.head.appendChild(style);

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createYaozarrsAnimation);
} else {
    createYaozarrsAnimation();
}

// Re-initialize when navigating in MkDocs (for SPA behavior)
document.addEventListener('DOMContentLoaded', createYaozarrsAnimation);