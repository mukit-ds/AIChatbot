import json
import uuid
import requests
import streamlit as st
from typing import Any, Dict, List, Generator

# -------------------------
# Config
# -------------------------
API_BASE = "http://127.0.0.1:9000"
STREAM_URL = f"{API_BASE}/chat/stream"

st.set_page_config(page_title="Marrfa AI", page_icon="🏠", layout="wide")

# -------------------------
# CSS (no sidebar + sticky input + carousel)
# -------------------------
st.markdown(
    """
<style>
header, footer { visibility: hidden; height: 0; }

section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="collapsedControl"] { display: none !important; }

.block-container { max-width: 1200px; padding-top: 26px; padding-bottom: 120px; }

/* bubbles */
.msg-wrap { display:flex; justify-content:flex-start; margin: 10px 0 14px 0; }
.msg-bubble {
    background: #F3F4F6;
    color: #111827;
    padding: 14px 18px;
    border-radius: 22px;
    width: min(880px, 92vw);
    box-shadow: 0 1px 0 rgba(0,0,0,0.04);
    font-size: 15px;
    line-height: 1.5;
}
.user-bubble { background: #EEF2F6; }

/* loading indicator */
.loading-bubble {
    background: #F3F4F6;
    color: #6B7280;
    padding: 14px 18px;
    border-radius: 22px;
    width: min(880px, 92vw);
    box-shadow: 0 1px 0 rgba(0,0,0,0.04);
    font-size: 15px;
    line-height: 1.5;
    display: flex;
    align-items: center;
    animation: pulse 1.5s ease-in-out infinite;
}

.dot {
    height: 8px;
    width: 8px;
    background-color: #6B7280;
    border-radius: 50%;
    display: inline-block;
    margin: 0 2px;
    animation: bounce 1.4s infinite ease-in-out both;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1.0); }
}

@keyframes pulse {
    0% { opacity: 0.8; }
    50% { opacity: 1; }
    100% { opacity: 0.8; }
}

/* property card */
.prop-card { 
    background: white; 
    border-radius: 18px; 
    border: 1px solid #EEE; 
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(0,0,0,0.04); 
    margin-bottom: 10px; 
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.prop-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* Image carousel */
.image-carousel {
    position: relative;
    width: 100%;
    height: 200px;
    overflow: hidden;
    border-radius: 10px 10px 0 0;
}

.carousel-image {
    width: 100%;
    height: 200px;
    object-fit: cover;
    display: block;
    transition: opacity 0.5s ease;
}

.carousel-dots {
    position: absolute;
    bottom: 10px;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    gap: 6px;
    z-index: 10;
}

.carousel-dot {
    width: 8px;
    height: 8px;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 50%;
    cursor: pointer;
    transition: background 0.3s ease, transform 0.3s ease;
}

.carousel-dot.active {
    background: white;
    transform: scale(1.2);
}

.carousel-nav {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(0, 0, 0, 0.3);
    color: white;
    border: none;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 10;
    transition: background 0.3s ease;
}

.carousel-nav:hover {
    background: rgba(0, 0, 0, 0.5);
}

.carousel-prev {
    left: 10px;
}

.carousel-next {
    right: 10px;
}

.carousel-nav.hidden {
    display: none;
}

.prop-body { padding: 14px 14px 6px 14px; }
.prop-title { 
    font-size: 18px; 
    font-weight: 800; 
    margin: 0; 
    color: #111827; 
    line-height: 1.3;
    margin-bottom: 8px;
}
.prop-description {
    color: #4B5563;
    font-size: 14px;
    line-height: 1.4;
    margin-bottom: 12px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.meta-box { 
    margin-top: 10px; 
    background: #F3F4F6; 
    border-radius: 14px; 
    padding: 10px 12px; 
}
.meta-line { 
    display: flex; 
    gap: 10px; 
    align-items: flex-start; 
    color: #374151; 
    font-size: 14px; 
    margin: 8px 0; 
}
.meta-icon { 
    width: 18px; 
    display: inline-block; 
    opacity: 0.9; 
    flex-shrink: 0;
}
.small-muted { 
    color: #6B7280; 
    font-size: 13px; 
    margin-bottom: 10px;
}

/* View details button */
.view-details-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
    margin-top: 8px;
    text-align: center;
}

.view-details-btn:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

/* sticky input */
.sticky-wrap{
  position: fixed; 
  left: 0; 
  right: 0; 
  bottom: 0;
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(10px);
  border-top: 1px solid #E5E7EB;
  padding: 14px 0;
  z-index: 9999;
  box-shadow: 0 -2px 20px rgba(0,0,0,0.05);
}
.sticky-inner{ 
    max-width: 1200px; 
    margin: 0 auto; 
    padding: 0 16px; 
}
.sticky-row{ 
    display: flex; 
    gap: 10px; 
    align-items: center; 
    background: white; 
    border: 1px solid #D1D5DB;
    border-radius: 14px; 
    padding: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.sticky-inner .stTextInput > div > div > input{
  border-radius: 12px !important;
  height: 44px !important;
  padding: 10px 12px !important;
  font-size: 16px !important;
  border: none !important;
  background: #F3F4F6 !important;
  color: #111827 !important;
}
.send-btn button{
  width: 44px !important; 
  height: 44px !important;
  border-radius: 10px !important;
  border: 1px solid #D1D5DB !important;
  background: white !important;
  font-size: 18px !important;
  padding: 0 !important;
  transition: all 0.2s ease;
}
.send-btn button:hover { 
    border-color: #9CA3AF !important; 
    background: #F3F4F6 !important;
}

/* Property image counter */
.image-counter {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(0, 0, 0, 0.6);
    color: white;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 12px;
    z-index: 10;
}

/* Empty image placeholder */
.empty-img {
    width: 100%;
    height: 200px;
    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6B7280;
    font-size: 14px;
    border-radius: 10px 10px 0 0;
}
</style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# State
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "selected_raw" not in st.session_state:
    st.session_state.selected_raw = {}
if "selected_title" not in st.session_state:
    st.session_state.selected_title = ""
if "carousel_states" not in st.session_state:
    st.session_state.carousel_states = {}


# -------------------------
# Helpers
# -------------------------
def safe_list(x):
    return x if isinstance(x, list) else []


def escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def fmt_price(raw: Dict[str, Any]) -> str:
    min_aed = raw.get("min_price_aed") or raw.get("min_price")
    max_aed = raw.get("max_price_aed") or raw.get("max_price")
    cur = raw.get("price_currency") or "AED"

    try:
        if min_aed and max_aed:
            min_val = float(min_aed) if isinstance(min_aed, (int, float, str)) else 0
            max_val = float(max_aed) if isinstance(max_aed, (int, float, str)) else 0
            return f"{min_val:,.0f} – {max_val:,.0f} {cur}"
        if min_aed:
            min_val = float(min_aed) if isinstance(min_aed, (int, float, str)) else 0
            return f"from {min_val:,.0f} {cur}"
        if max_aed:
            max_val = float(max_aed) if isinstance(max_aed, (int, float, str)) else 0
            return f"up to {max_val:,.0f} {cur}"
    except:
        pass
    return "Price not available"


def fmt_completion(raw: Dict[str, Any], p: Dict[str, Any]) -> str:
    y = p.get("completion_year")
    if y:
        return str(y)
    dt = raw.get("completion_datetime") or raw.get("completion_date")
    if isinstance(dt, str) and len(dt) >= 10:
        return dt[:10]
    if isinstance(dt, str) and len(dt) >= 4:
        return dt[:4]
    return "N/A"


def sse_lines(resp: requests.Response) -> Generator[str, None, None]:
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        yield raw.strip()


def read_sse_events(resp: requests.Response) -> Generator[Dict[str, Any], None, None]:
    event_name = "message"
    data_buf = ""

    for line in sse_lines(resp):
        if not line:
            if data_buf:
                try:
                    payload = json.loads(data_buf)
                except Exception:
                    payload = {"type": "raw", "raw": data_buf}
                yield {"event": event_name, "data": payload}
            event_name = "message"
            data_buf = ""
            continue

        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            chunk = line.split(":", 1)[1].strip()
            data_buf += chunk

    # Handle any remaining data
    if data_buf:
        try:
            payload = json.loads(data_buf)
        except Exception:
            payload = {"type": "raw", "raw": data_buf}
        yield {"event": event_name, "data": payload}


def render_image_carousel(images: List[str], carousel_id: str) -> str:
    """Render HTML for image carousel with navigation"""
    if not images:
        return '<div class="empty-img">No images available</div>'

    images = images[:5]  # Limit to 5 images

    carousel_html = f'''
    <div class="image-carousel" id="carousel-{carousel_id}">
        <div class="image-counter">{len(images)} image{"" if len(images) == 1 else "s"}</div>
    '''

    # Add images
    for i, img_url in enumerate(images):
        display_style = "block" if i == 0 else "none"
        carousel_html += f'<img class="carousel-image" src="{escape_html(img_url)}" style="display: {display_style};" data-index="{i}">'

    # Add navigation buttons if multiple images
    if len(images) > 1:
        carousel_html += '''
        <button class="carousel-nav carousel-prev" onclick="prevImage('carousel-''' + carousel_id + '''')">‹</button>
        <button class="carousel-nav carousel-next" onclick="nextImage('carousel-''' + carousel_id + '''')">›</button>
        <div class="carousel-dots">
        '''

        for i in range(len(images)):
            active_class = "active" if i == 0 else ""
            carousel_html += f'<div class="carousel-dot {active_class}" onclick="goToImage(\'carousel-{carousel_id}\', {i})"></div>'

        carousel_html += '</div>'

    carousel_html += '</div>'

    return carousel_html


def get_property_description(p: Dict[str, Any], raw: Dict[str, Any]) -> str:
    """Extract or generate property description"""
    # Try to get description from various fields
    description = p.get("description") or raw.get("description") or raw.get("short_description") or ""

    if not description:
        # Generate a generic description based on available data
        title = p.get("title", "This property")
        location = p.get("location", "Dubai")
        developer = p.get("developer", "a reputable developer")
        completion = fmt_completion(raw, p)

        # Check for amenities
        amenities = []
        amenity_fields = {
            "has_pool": "swimming pool",
            "has_gym": "fitness center",
            "has_parking": "parking facilities",
            "has_security": "24/7 security",
            "has_garden": "landscaped gardens",
            "has_play_area": "children's play area",
            "has_concierge": "concierge service"
        }

        for field, amenity_name in amenity_fields.items():
            if raw.get(field):
                amenities.append(amenity_name)

        amenity_text = ""
        if amenities:
            if len(amenities) == 1:
                amenity_text = f" featuring a {amenities[0]}."
            elif len(amenities) == 2:
                amenity_text = f" featuring {amenities[0]} and {amenities[1]}."
            else:
                amenity_text = f" featuring {', '.join(amenities[:-1])}, and {amenities[-1]}."

        # Get property type
        prop_type = raw.get("unit_types") or raw.get("property_type") or ""
        if isinstance(prop_type, list):
            prop_type = prop_type[0] if prop_type else ""

        # Generate description
        if prop_type:
            description = f"{title} is a {prop_type.lower()} located in {location}, developed by {developer}. With completion scheduled for {completion}, it offers modern living spaces with contemporary designs and high-quality finishes.{amenity_text} This property represents an excellent investment opportunity in one of Dubai's most sought-after neighborhoods."
        else:
            description = f"{title} is a premium residential development located in {location}, developed by {developer}. With completion scheduled for {completion}, it offers modern living spaces with contemporary designs and high-quality finishes.{amenity_text} This property represents an excellent investment opportunity in one of Dubai's most sought-after neighborhoods."

    # Truncate if too long
    if len(description) > 200:
        description = description[:197] + "..."

    return description


def stream_chat(query: str) -> Dict[str, Any]:
    payload = {"query": query, "session_id": st.session_state.session_id, "is_logged_in": False}

    # Create placeholders
    assistant_placeholder = st.empty()
    loading_placeholder = st.empty()

    # Clear previous loading if exists
    loading_placeholder.empty()

    # Show initial loading indicator immediately
    loading_placeholder.markdown(
        """
<div class="msg-wrap">
  <div class="loading-bubble">
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
    <span style="margin-left: 10px;">Processing your request...</span>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )

    typing_text = ""
    final_payload: Dict[str, Any] = {}

    try:
        with requests.post(STREAM_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()

            for ev in read_sse_events(resp):
                data = ev["data"]

                if data.get("type") == "loading":
                    # Update loading message
                    loading_placeholder.markdown(
                        f"""
<div class="msg-wrap">
  <div class="loading-bubble">
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
    <span style="margin-left: 10px;">{escape_html(data.get("message", "Processing..."))}</span>
  </div>
</div>
                        """,
                        unsafe_allow_html=True
                    )

                elif data.get("type") == "delta":
                    # Hide loading indicator when we start getting content
                    loading_placeholder.empty()

                    typing_text += data.get("delta", "")
                    html = escape_html(typing_text).replace("\n", "<br>")
                    assistant_placeholder.markdown(
                        f"""
<div class="msg-wrap">
  <div class="msg-bubble">{html}</div>
</div>
                        """,
                        unsafe_allow_html=True
                    )

                elif data.get("type") == "content_start":
                    # Hide loading when content starts
                    loading_placeholder.empty()

                elif data.get("type") == "final":
                    # Hide any remaining loading indicator
                    loading_placeholder.empty()

                    final_payload = data
                    typing_text = final_payload.get("reply", typing_text)
                    html = escape_html(typing_text).replace("\n", "<br>")

                    # Update the final message
                    assistant_placeholder.markdown(
                        f"""
<div class="msg-wrap">
  <div class="msg-bubble">{html}</div>
</div>
                        """,
                        unsafe_allow_html=True
                    )

                elif data.get("type") == "error":
                    loading_placeholder.empty()
                    error_msg = data.get("message", "An error occurred")
                    assistant_placeholder.markdown(
                        f"""
<div class="msg-wrap">
  <div class="msg-bubble" style="color: #d32f2f;">{escape_html(error_msg)}</div>
</div>
                        """,
                        unsafe_allow_html=True
                    )
                    final_payload = {"reply": error_msg, "properties": [], "properties_full": [], "total": 0}

                elif data.get("type") == "done":
                    break

    except requests.exceptions.ConnectionError as e:
        loading_placeholder.empty()
        err = "❌ Connection error: Cannot connect to the server. Please make sure the backend is running."
        assistant_placeholder.markdown(
            f"""
<div class="msg-wrap">
  <div class="msg-bubble" style="color: #d32f2f;">{escape_html(err)}</div>
</div>
            """,
            unsafe_allow_html=True
        )
        return {"reply": err, "properties": [], "properties_full": [], "total": 0}
    except requests.exceptions.RequestException as e:
        loading_placeholder.empty()
        err = f"❌ Network error: {str(e)}"
        assistant_placeholder.markdown(
            f"""
<div class="msg-wrap">
  <div class="msg-bubble" style="color: #d32f2f;">{escape_html(err)}</div>
</div>
            """,
            unsafe_allow_html=True
        )
        return {"reply": err, "properties": [], "properties_full": [], "total": 0}
    except Exception as e:
        loading_placeholder.empty()
        err = f"❌ Unexpected error: {str(e)}"
        assistant_placeholder.markdown(
            f"""
<div class="msg-wrap">
  <div class="msg-bubble" style="color: #d32f2f;">{escape_html(err)}</div>
</div>
            """,
            unsafe_allow_html=True
        )
        return {"reply": err, "properties": [], "properties_full": [], "total": 0}

    if not final_payload:
        final_payload = {"reply": typing_text, "properties": [], "properties_full": [], "total": 0}

    return final_payload


# -------------------------
# Carousel JavaScript
# -------------------------
carousel_js = """
<script>
function showImage(carouselId, index) {
    const carousel = document.getElementById(carouselId);
    if (!carousel) return;

    const images = carousel.querySelectorAll('.carousel-image');
    const dots = carousel.querySelectorAll('.carousel-dot');

    images.forEach(img => img.style.display = 'none');
    dots.forEach(dot => dot.classList.remove('active'));

    if (images[index]) {
        images[index].style.display = 'block';
    }
    if (dots[index]) {
        dots[index].classList.add('active');
    }
}

function nextImage(carouselId) {
    const carousel = document.getElementById(carouselId);
    if (!carousel) return;

    const images = carousel.querySelectorAll('.carousel-image');
    const currentIndex = Array.from(images).findIndex(img => img.style.display === 'block');
    const nextIndex = (currentIndex + 1) % images.length;
    showImage(carouselId, nextIndex);
}

function prevImage(carouselId) {
    const carousel = document.getElementById(carouselId);
    if (!carousel) return;

    const images = carousel.querySelectorAll('.carousel-image');
    const currentIndex = Array.from(images).findIndex(img => img.style.display === 'block');
    const prevIndex = (currentIndex - 1 + images.length) % images.length;
    showImage(carouselId, prevIndex);
}

function goToImage(carouselId, index) {
    showImage(carouselId, index);
}

// Auto-rotate carousels
document.addEventListener('DOMContentLoaded', function() {
    const carousels = document.querySelectorAll('.image-carousel');

    carousels.forEach(carousel => {
        const images = carousel.querySelectorAll('.carousel-image');
        if (images.length > 1) {
            let currentIndex = 0;

            // Auto-rotate every 5 seconds
            const intervalId = setInterval(() => {
                currentIndex = (currentIndex + 1) % images.length;
                showImage(carousel.id, currentIndex);
            }, 5000);

            // Stop auto-rotation on hover
            carousel.addEventListener('mouseenter', () => {
                clearInterval(intervalId);
            });

            // Resume auto-rotation when mouse leaves
            carousel.addEventListener('mouseleave', () => {
                clearInterval(intervalId);
                setInterval(() => {
                    currentIndex = (currentIndex + 1) % images.length;
                    showImage(carousel.id, currentIndex);
                }, 5000);
            });
        }
    });
});
</script>
"""

# -------------------------
# Render conversation
# -------------------------
if not st.session_state.history:
    st.markdown(
        """
<div class="msg-wrap">
  <div class="msg-bubble">👋 Welcome to Marrfa AI! Ask about properties, Marrfa company info, or search for specific developments in Dubai.</div>
</div>
        """,
        unsafe_allow_html=True
    )
else:
    for item in st.session_state.history:
        q = item.get("q", "")
        if q:
            q_html = escape_html(q)
            st.markdown(
                f"""
<div class="msg-wrap">
  <div class="msg-bubble user-bubble">{q_html}</div>
</div>
                """,
                unsafe_allow_html=True
            )

        reply = item.get("reply", "")
        reply_html = escape_html(reply).replace("\n", "<br>")
        st.markdown(
            f"""
<div class="msg-wrap">
  <div class="msg-bubble">{reply_html}</div>
</div>
            """,
            unsafe_allow_html=True
        )

        data = item.get("data", {}) or {}
        props = safe_list(data.get("properties"))
        props_full = safe_list(data.get("properties_full"))

        if props:
            total_count = data.get("total", len(props))
            st.markdown(
                f'<div class="small-muted" style="margin: 6px 0 14px 0;">Showing {len(props)} of {total_count} result{"" if total_count == 1 else "s"}</div>',
                unsafe_allow_html=True
            )

            # Use 3-column layout for property cards
            cols = st.columns(3, gap="large")

            for i, p in enumerate(props):
                col = cols[i % 3]
                raw = props_full[i] if i < len(props_full) else {}

                # Extract property data
                title = p.get("title") or raw.get("name") or "Untitled Property"
                location = p.get("location") or raw.get("area") or "Dubai"
                developer = p.get("developer") or raw.get("developer") or "Not specified"
                completion = fmt_completion(raw, p)
                price_text = fmt_price(raw)
                cover = p.get("cover_image") or raw.get("cover_image_url")

                # Get additional images (if available)
                additional_images = p.get("property_images") or []
                if cover and cover not in additional_images:
                    all_images = [cover] + additional_images
                else:
                    all_images = additional_images if additional_images else []

                # Get property description
                description = get_property_description(p, raw)

                with col:
                    # Render image carousel or single image
                    carousel_id = f"{item.get('id', 'x')}_{i}"

                    if len(all_images) > 1:
                        # Multiple images - use carousel
                        img_html = render_image_carousel(all_images[:5], carousel_id)
                    elif cover:
                        # Single image
                        img_html = f'<img class="carousel-image" src="{escape_html(cover)}" style="width: 100%; height: 200px; object-fit: cover; display: block;">'
                    else:
                        # No images
                        img_html = '<div class="empty-img">No images available</div>'

                    # Render property card
                    st.markdown(
                        f"""
<div class="prop-card">
  {img_html}
  <div class="prop-body">
    <div class="prop-title">{escape_html(title)}</div>
    <div class="prop-description" title="{escape_html(description)}">{escape_html(description)}</div>
    <div class="meta-box">
      <div class="meta-line">
        <span class="meta-icon">📍</span>
        <span>{escape_html(location)}</span>
      </div>
      <div class="meta-line">
        <span class="meta-icon">🏗️</span>
        <span>{escape_html(developer)}</span>
      </div>
      <div class="meta-line">
        <span class="meta-icon">🗓️</span>
        <span>Completion: {escape_html(str(completion))}</span>
      </div>
      <div class="meta-line">
        <span class="meta-icon">💰</span>
        <span>{escape_html(str(price_text))}</span>
      </div>
    </div>
    <button class="view-details-btn" onclick="(function(){{window.open('{p.get("listing_url", "#")}', '_blank');}})()">
      View Details
    </button>
  </div>
</div>
                        """,
                        unsafe_allow_html=True
                    )

# Inject carousel JavaScript
st.markdown(carousel_js, unsafe_allow_html=True)

# Details panel for selected property
if st.session_state.selected_raw:
    st.markdown("---")
    st.subheader(f"Property Details: {st.session_state.selected_title}")
    st.json(st.session_state.selected_raw)

# -------------------------
# Sticky input bottom
# -------------------------
st.markdown('<div class="sticky-wrap"><div class="sticky-inner"><div class="sticky-row">', unsafe_allow_html=True)

with st.form("sticky_form", clear_on_submit=True):
    c1, c2 = st.columns([12, 1], gap="small")

    with c1:
        prompt = st.text_input(
            "Message",
            placeholder="Ask about properties, Marrfa company info, or search for specific developments...",
            label_visibility="collapsed",
        )

    with c2:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        submitted = st.form_submit_button("▶")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted and prompt.strip():
        st.session_state.history.append({
            "id": str(uuid.uuid4())[:8],
            "q": prompt.strip(),
            "reply": "",
            "data": {}
        })
        st.rerun()

st.markdown("</div></div></div>", unsafe_allow_html=True)

# -------------------------
# After rerun: if last message reply empty -> stream now
# -------------------------
if st.session_state.history and st.session_state.history[-1].get("reply", "") == "":
    last = st.session_state.history[-1]
    q = last["q"]

    final = stream_chat(q)

    last["reply"] = final.get("reply", "")
    last["data"] = final

    st.session_state.history[-1] = last
    st.rerun()