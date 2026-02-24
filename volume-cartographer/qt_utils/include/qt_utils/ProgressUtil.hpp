#pragma once

#include <QFont>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QTimer>

class QProgressBar;
class QStatusBar;

namespace qt_utils {

/// Text display mode for the progress bar
enum class ProgressTextMode {
    Percent,        ///< Show "42%"
    Fraction,       ///< Show "42 / 100"
    DoneRemaining,  ///< Show "42 done : 58 left"
    Custom,         ///< User-defined format string
    None            ///< No text
};

/// Configuration for how the progress bar renders text
struct ProgressBarOptions {
    ProgressTextMode textMode = ProgressTextMode::Percent;
    QString prefix;
    QString suffix;
    QString customFormat;  ///< Supports {value}, {total}, {remaining}, {percent}
    int percentPrecision = 0;
    int fontPointSize = -1;  ///< -1 means use the bar's existing font size
};

/// Status-bar progress management utility.
///
/// Provides two independent capabilities that can be used together:
///   - An animated spinner (busy indicator) shown in the status bar message area.
///   - A managed QProgressBar with configurable text formatting, 0-100%
///     progress, and busy-indicator mode (total <= 0).
///
/// The utility does not own the QStatusBar or QProgressBar; it merely drives
/// them.  When a progress bar is assigned via setProgressBar(), the original
/// font, format, and range are captured so they can be restored on
/// stopProgress().
class ProgressUtil : public QObject {
    Q_OBJECT

public:
    /// Construct a ProgressUtil that drives the given status bar.
    explicit ProgressUtil(QStatusBar* statusBar, QObject* parent = nullptr);
    ~ProgressUtil() override;

    // -- Animated spinner ---------------------------------------------------

    /// Start a spinning animation in the status bar message area.
    void startAnimation(const QString& message);

    /// Stop the animation and show a final message.
    /// @param timeout  How long to display the message in ms (0 = indefinite).
    void stopAnimation(const QString& message, int timeout = 15000);

    // -- Progress bar -------------------------------------------------------

    /// Associate a progress bar widget for this util to manage.
    void setProgressBar(QProgressBar* progressBar);

    /// Set default options used whenever startProgress() is called without
    /// an explicit override.
    void configureProgressBar(const ProgressBarOptions& options);

    /// Begin tracking progress.
    /// @param totalSteps  Expected step count (<= 0 activates the busy
    ///                    indicator).
    /// @param options     Optional override of the default bar options.
    void startProgress(int totalSteps, const ProgressBarOptions* options = nullptr);

    /// Set the current progress to an absolute step count.
    void updateProgress(int completedSteps);

    /// Increment progress by @p stepDelta (default 1).
    void advanceProgress(int stepDelta = 1);

    /// End progress tracking and optionally reset the bar value.
    void stopProgress(bool resetValue = true);

    // -- Queries ------------------------------------------------------------

    [[nodiscard]] auto isProgressActive() const -> bool;
    [[nodiscard]] auto progressTotal() const -> int;
    [[nodiscard]] auto progressValue() const -> int;

private slots:
    void onAnimationTick();

private:
    void applyProgressBarFont();
    void restoreProgressBarFont();
    void restoreProgressBarFormat();
    void updateProgressFormat();

    // Spinner state
    QStatusBar* statusBar_{nullptr};
    QTimer* animTimer_{nullptr};
    int animFrame_{0};
    QString animMessage_;

    // Progress bar state
    QPointer<QProgressBar> progressBar_;
    ProgressBarOptions defaultOptions_;
    ProgressBarOptions activeOptions_;
    int progressTotal_{0};
    int progressValue_{0};
    bool progressActive_{false};

    // Saved originals from the bar so we can restore on stop
    bool hasStoredFont_{false};
    bool hasStoredFormat_{false};
    int storedMinimum_{0};
    int storedMaximum_{100};
    QString storedFormat_;
    QFont storedFont_;
};

}  // namespace qt_utils
