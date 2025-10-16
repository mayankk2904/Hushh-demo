// This is the main entry point for the application. It imports and renders the main components of the app.
import Header from "./components/Header";
//import HeroSection from "./components/HeroSection"; // Ensure the file exists at this path or update the path accordingly.
import DashboardPreview from "./components/DashboardPreview";
import FeaturesSection from "./components/Features";
import AboutSection from "./components/AboutTE";
import Footer from "./components/Footer";
import HeroSection from "./components/HeroSection"; // Ensure the file exists at this path or update the path accordingly.
export default function Home() {
  return (
    <div className="min-h-screen bg-[#fad2ad]">
      <Header />
      <HeroSection/>
      <DashboardPreview />
      <FeaturesSection />
      <AboutSection />
      <Footer />
    </div>
  );
}